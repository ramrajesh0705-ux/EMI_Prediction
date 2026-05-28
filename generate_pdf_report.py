from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def create_pdf_comparison():
    doc = SimpleDocTemplate("mlflow_comparison_report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        borderWidth=0,
        borderColor=colors.blue,
        borderPadding=5
    )

    normal_style = styles['Normal']

    story = []

    # Title
    story.append(Paragraph("MLflow Runs Comparison Report", title_style))
    story.append(Paragraph("<b>Generated on:</b> April 29, 2026", normal_style))
    story.append(Paragraph("<b>Project:</b> EMI Prediction App - FinTech ML Suite", normal_style))
    story.append(Spacer(1, 0.5*inch))

    # Classification Section
    story.append(Paragraph("Classification Experiments (EMI Eligibility Prediction)", heading_style))

    # Classification table data
    classification_data = [
        ['Experiment Name', 'Model Type', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC Macro'],
        ['EMI Eligibility Prediction Logistic Regression version 2', 'Logistic Regression v2', '0.7647', '0.6124', '0.6284', '0.7283', '0.9160'],
        ['EMI Eligibility Prediction Logistic Regression Experiment version 1', 'Logistic Regression v1', '0.8973', '0.5825', '0.5697', '0.5959', '0.9339'],
        ['EMI Eligibility Prediction Random Forrest Classification experiment', 'Random Forest Classifier', '0.8088', '0.6475', '0.6433', '0.7560', 'N/A'],
        ['EMI Eligibility Prediction XG Boost Classification experiment', 'XGBoost Classifier', '0.8943', '0.7310', '0.7209', '0.7966', 'N/A']
    ]

    classification_table = Table(classification_data)
    classification_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(classification_table)
    story.append(Spacer(1, 0.3*inch))

    # Classification Summary
    story.append(Paragraph("Classification Summary", styles['Heading3']))
    summary_text = """
    • Best Accuracy: Logistic Regression v1 (0.8973)<br/>
    • Best F1 Score: XGBoost Classifier (0.7310)<br/>
    • Best ROC AUC: Logistic Regression v1 (0.9339)<br/>
    • Logistic Regression models show strong ROC AUC performance, indicating good probability calibration<br/>
    • XGBoost provides the best balance of precision and recall
    """
    story.append(Paragraph(summary_text, normal_style))
    story.append(Spacer(1, 0.5*inch))

    # Regression Section
    story.append(Paragraph("Regression Experiments (Max Monthly EMI Prediction)", heading_style))

    # Regression table data
    regression_data = [
        ['Experiment Name', 'Model Type', 'R² Score', 'MAE', 'MAPE', 'MSE'],
        ['Max Monthly EMI Prediction Random Forest Regressor experiment', 'Random Forest Regressor', '0.9608', '0.1653', '0.0213', '0.0727'],
        ['Max Monthly EMI Prediction XGBoost Regressor experiment', 'XGBoost Regressor', '0.9438', '0.2282', '0.0312', '0.1043'],
        ['Max Monthly EMI Prediction Linear Regression experiment', 'Linear Regression', '0.7587', '0.5089', '0.0683', '0.4478']
    ]

    regression_table = Table(regression_data)
    regression_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    story.append(regression_table)
    story.append(Spacer(1, 0.3*inch))

    # Regression Summary
    story.append(Paragraph("Regression Summary", styles['Heading3']))
    regression_summary = """
    • Best R² Score: Random Forest Regressor (0.9608) - explains 96.08% of variance<br/>
    • Best MAE: Random Forest Regressor (0.1653) - lowest absolute error<br/>
    • Best MAPE: Random Forest Regressor (0.0213) - 2.13% mean absolute percentage error<br/>
    • Random Forest significantly outperforms other models in all regression metrics<br/>
    • Linear Regression shows the poorest performance, indicating non-linear relationships in the data
    """
    story.append(Paragraph(regression_summary, normal_style))
    story.append(Spacer(1, 0.5*inch))

    # Key Insights
    story.append(Paragraph("Key Insights from All Runs", heading_style))
    insights = """
    1. Model Selection Recommendations:<br/>
       • Classification: XGBoost Classifier for balanced performance (best F1 score)<br/>
       • Regression: Random Forest Regressor for highest accuracy and lowest error<br/><br/>
    2. Performance Trends:<br/>
       • Tree-based models (Random Forest, XGBoost) consistently outperform linear models<br/>
       • Ensemble methods provide better generalization for both tasks<br/>
       • Logistic Regression shows good probability calibration (high ROC AUC) despite lower accuracy<br/><br/>
    3. Experiment Distribution:<br/>
       • 4 classification experiments focused on EMI eligibility prediction<br/>
       • 3 regression experiments focused on maximum EMI amount prediction<br/>
       • All experiments used proper MLflow tracking with comprehensive metrics logging<br/><br/>
    4. Database vs File System:<br/>
       • The mlflow.db contains minimal experiment metadata (only 2 experiments logged)<br/>
       • Full run details are stored in the file system under mlruns/ directories<br/>
       • File-based storage provides more detailed metrics and parameters than the database
    """
    story.append(Paragraph(insights, normal_style))

    # Footer
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("This comparison demonstrates a thorough model development process with multiple algorithms tested for both classification and regression tasks in the EMI prediction domain.", normal_style))

    doc.build(story)
    print("PDF report generated successfully!")

if __name__ == "__main__":
    create_pdf_comparison()