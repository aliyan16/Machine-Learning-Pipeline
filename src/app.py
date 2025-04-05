from flask import Flask, render_template
import json
import os

app = Flask(__name__)

def load_metrics():
    """Load metrics from the JSON file"""
    metrics_path = 'reports/metrics.json'
    if not os.path.exists(metrics_path):
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

@app.route('/')
def display_metrics():
    """Display the evaluation metrics on the webpage"""
    metrics = load_metrics()
    
    if metrics is None:
        return render_template('error.html', message="Metrics not found. Please run evaluation first.")
    
    return render_template('metrics.html', 
                          accuracy=metrics['accuracy'],
                          precision=metrics['precision'],
                          recall=metrics['recall'],
                          auc=metrics['auc'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)