"""
Results Export & Report Generation Module
Creates comprehensive reports in PDF, Excel, and CSV formats
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Optional, Any
import pickle
import json

# For PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. PDF generation will be limited.")


class ResultsExporter:
    """Export and report generation for disease gene detection results"""
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize results exporter
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def create_gene_impact_table(self, 
                                 gene_names: List[str],
                                 importance_scores: np.ndarray,
                                 predictions: np.ndarray,
                                 prediction_proba: np.ndarray,
                                 class_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create table of high-impact genes with disease classification probabilities
        
        Args:
            gene_names: List of gene names
            importance_scores: Feature importance scores
            predictions: Predicted classes
            prediction_proba: Prediction probabilities
            class_names: Optional class names
            
        Returns:
            DataFrame with gene impact information
        """
        n_classes = prediction_proba.shape[1] if len(prediction_proba.shape) > 1 else 1
        
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(n_classes)]
        
        # Create base table
        results = {
            'Gene_Name': gene_names,
            'Importance_Score': importance_scores,
            'Rank': range(1, len(gene_names) + 1)
        }
        
        # Add probability columns for each class
        if len(prediction_proba.shape) > 1:
            for i, class_name in enumerate(class_names):
                # Average probability across samples for each gene
                results[f'{class_name}_Probability'] = [
                    prediction_proba[:, i].mean() for _ in gene_names
                ]
        
        df = pd.DataFrame(results)
        
        # Sort by importance score
        df = df.sort_values('Importance_Score', ascending=False)
        df['Rank'] = range(1, len(df) + 1)
        
        return df
    
    def export_to_csv(self, data: pd.DataFrame, filename: str):
        """
        Export results to CSV
        
        Args:
            data: DataFrame to export
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"✅ Exported to CSV: {filepath}")
        return filepath
    
    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame], filename: str):
        """
        Export multiple DataFrames to Excel with multiple sheets
        
        Args:
            data_dict: Dictionary of sheet_name: DataFrame
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Exported to Excel: {filepath}")
        return filepath
    
    def create_summary_stats(self, 
                            model_metrics: Dict,
                            feature_selection: Dict,
                            preprocessing_stats: Dict) -> pd.DataFrame:
        """
        Create summary statistics table
        
        Args:
            model_metrics: Model evaluation metrics
            feature_selection: Feature selection results
            preprocessing_stats: Preprocessing statistics
            
        Returns:
            Summary DataFrame
        """
        summary = {
            'Category': [],
            'Metric': [],
            'Value': []
        }
        
        # Preprocessing stats
        summary['Category'].append('Preprocessing')
        summary['Metric'].append('Original Samples')
        summary['Value'].append(preprocessing_stats.get('original_shape', ['N/A', 'N/A'])[0])
        
        summary['Category'].append('Preprocessing')
        summary['Metric'].append('Final Samples')
        summary['Value'].append(preprocessing_stats.get('cleaned_shape', ['N/A', 'N/A'])[0])
        
        summary['Category'].append('Preprocessing')
        summary['Metric'].append('Missing Values Handled')
        summary['Value'].append(preprocessing_stats.get('missing_values_handled', 'N/A'))
        
        # Feature selection
        summary['Category'].append('Feature Selection')
        summary['Metric'].append('Original Features')
        summary['Value'].append(feature_selection.get('original_features', 'N/A'))
        
        summary['Category'].append('Feature Selection')
        summary['Metric'].append('Selected Features')
        summary['Value'].append(feature_selection.get('selected_features', 'N/A'))
        
        # Model performance
        for metric_name, metric_value in model_metrics.items():
            if isinstance(metric_value, (int, float)):
                summary['Category'].append('Model Performance')
                summary['Metric'].append(metric_name.replace('_', ' ').title())
                summary['Value'].append(f'{metric_value:.4f}' if isinstance(metric_value, float) else metric_value)
        
        df = pd.DataFrame(summary)
        return df
    
    def generate_pdf_report(self,
                           gene_impact_table: pd.DataFrame,
                           model_comparison: pd.DataFrame,
                           summary_stats: pd.DataFrame,
                           confusion_matrix: np.ndarray,
                           image_paths: Optional[List[str]] = None,
                           filename: str = 'disease_gene_report.pdf'):
        """
        Generate comprehensive PDF report
        
        Args:
            gene_impact_table: Gene impact DataFrame
            model_comparison: Model comparison DataFrame
            summary_stats: Summary statistics DataFrame
            confusion_matrix: Confusion matrix
            image_paths: List of image file paths to include
            filename: Output filename
        """
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab not installed. Cannot generate PDF.")
            print("Install with: pip install reportlab")
            return None
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E4053'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        story.append(Paragraph("Disease Gene Detection Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Report info
        report_info = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Analysis Type:</b> Gene Expression Classification<br/>
        <b>Report ID:</b> {self.timestamp}
        """
        story.append(Paragraph(report_info, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Section 1: Summary Statistics
        story.append(Paragraph("1. Summary Statistics", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        summary_data = [summary_stats.columns.tolist()] + summary_stats.values.tolist()
        summary_table = Table(summary_data, colWidths=[2*inch, 2.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Section 2: Model Comparison
        story.append(Paragraph("2. Model Performance Comparison", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        model_data = [model_comparison.columns.tolist()] + model_comparison.head(10).values.tolist()
        model_table = Table(model_data)
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(model_table)
        story.append(PageBreak())
        
        # Section 3: High-Impact Genes
        story.append(Paragraph("3. High-Impact Genes", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        gene_data = [gene_impact_table.columns.tolist()] + gene_impact_table.head(20).values.tolist()
        gene_table = Table(gene_data)
        gene_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(gene_table)
        story.append(PageBreak())
        
        # Section 4: Visualizations
        if image_paths:
            story.append(Paragraph("4. Visualizations", styles['Heading2']))
            story.append(Spacer(1, 0.2*inch))
            
            for img_path in image_paths:
                if os.path.exists(img_path):
                    try:
                        img = Image(img_path, width=5*inch, height=3.5*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        print(f"Warning: Could not add image {img_path}: {e}")
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF Report generated: {filepath}")
        return filepath
    
    def export_complete_results(self,
                               gene_impact_table: pd.DataFrame,
                               model_comparison: pd.DataFrame,
                               model_metrics: Dict,
                               feature_selection: Dict,
                               preprocessing_stats: Dict,
                               confusion_matrix: np.ndarray,
                               image_paths: Optional[List[str]] = None,
                               base_filename: str = 'results'):
        """
        Export complete results in all formats
        
        Args:
            gene_impact_table: Gene impact table
            model_comparison: Model comparison DataFrame
            model_metrics: Model evaluation metrics
            feature_selection: Feature selection results
            preprocessing_stats: Preprocessing statistics
            confusion_matrix: Confusion matrix
            image_paths: List of visualization image paths
            base_filename: Base name for output files
        """
        print("\n" + "="*70)
        print("EXPORTING COMPLETE RESULTS")
        print("="*70)
        
        # Create summary stats
        summary_stats = self.create_summary_stats(
            model_metrics, feature_selection, preprocessing_stats
        )
        
        # 1. Export to CSV
        csv_files = {
            'gene_impact': self.export_to_csv(
                gene_impact_table, 
                f'{base_filename}_gene_impact_{self.timestamp}.csv'
            ),
            'model_comparison': self.export_to_csv(
                model_comparison,
                f'{base_filename}_model_comparison_{self.timestamp}.csv'
            ),
            'summary': self.export_to_csv(
                summary_stats,
                f'{base_filename}_summary_{self.timestamp}.csv'
            )
        }
        
        # 2. Export to Excel (all sheets in one file)
        excel_data = {
            'Summary': summary_stats,
            'Model Comparison': model_comparison,
            'Top Genes': gene_impact_table.head(100),
            'All Genes': gene_impact_table
        }
        
        excel_file = self.export_to_excel(
            excel_data,
            f'{base_filename}_complete_{self.timestamp}.xlsx'
        )
        
        # 3. Generate PDF Report
        if REPORTLAB_AVAILABLE:
            pdf_file = self.generate_pdf_report(
                gene_impact_table,
                model_comparison,
                summary_stats,
                confusion_matrix,
                image_paths,
                f'{base_filename}_report_{self.timestamp}.pdf'
            )
        else:
            pdf_file = None
        
        # 4. Export metadata as JSON
        metadata = {
            'timestamp': self.timestamp,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'preprocessing_stats': preprocessing_stats,
            'feature_selection': {k: str(v) for k, v in feature_selection.items()},
            'model_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                            for k, v in model_metrics.items()},
            'n_genes_analyzed': len(gene_impact_table),
            'n_models_compared': len(model_comparison)
        }
        
        metadata_file = os.path.join(self.output_dir, f'{base_filename}_metadata_{self.timestamp}.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✅ Metadata saved: {metadata_file}")
        
        print("\n" + "="*70)
        print("EXPORT COMPLETE")
        print("="*70)
        print(f"\n📁 All results saved to: {self.output_dir}/")
        print(f"\n📊 Files generated:")
        print(f"  • CSV files: {len(csv_files)}")
        print(f"  • Excel file: {excel_file}")
        print(f"  • PDF report: {pdf_file if pdf_file else 'Not generated (install reportlab)'}")
        print(f"  • Metadata: {metadata_file}")
        
        return {
            'csv_files': csv_files,
            'excel_file': excel_file,
            'pdf_file': pdf_file,
            'metadata_file': metadata_file
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Gene impact table
    gene_names = [f'GENE_{i+1}' for i in range(100)]
    importance_scores = np.random.rand(100)
    predictions = np.random.randint(0, 3, 100)
    prediction_proba = np.random.rand(100, 3)
    prediction_proba = prediction_proba / prediction_proba.sum(axis=1, keepdims=True)
    
    exporter = ResultsExporter(output_dir='results')
    
    gene_table = exporter.create_gene_impact_table(
        gene_names, importance_scores, predictions, prediction_proba,
        class_names=['Breast Cancer', 'Lung Cancer', 'Healthy']
    )
    
    # Model comparison
    model_comparison = pd.DataFrame({
        'Model': ['SVM', 'Random Forest', 'ANN', 'KNN'],
        'Accuracy': [0.85, 0.92, 0.88, 0.83],
        'Precision': [0.84, 0.91, 0.87, 0.82],
        'Recall': [0.83, 0.90, 0.86, 0.81],
        'F1-Score': [0.83, 0.90, 0.86, 0.81],
        'ROC-AUC': [0.88, 0.94, 0.90, 0.85]
    })
    
    # Export all results
    results = exporter.export_complete_results(
        gene_impact_table=gene_table,
        model_comparison=model_comparison,
        model_metrics={'accuracy': 0.92, 'precision': 0.91, 'recall': 0.90},
        feature_selection={'original_features': 500, 'selected_features': 100},
        preprocessing_stats={
            'original_shape': (1000, 500),
            'cleaned_shape': (980, 500),
            'missing_values_handled': 50
        },
        confusion_matrix=np.array([[85, 5, 3], [4, 90, 2], [2, 3, 95]]),
        base_filename='example_results'
    )
    
    print("\nExample results exported!")
