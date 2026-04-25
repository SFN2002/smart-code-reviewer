from __future__ import annotations
import os
import json
from typing import List, Dict, Any

class HTMLReporter:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(self, results: List[Dict[str, Any]], filename: str = "report.html"):
        path = os.path.join(self.output_dir, filename)
        anomalies = sum(1 for r in results if r.get("is_anomalous"))
        avg_confidence = sum(r.get("confidence", 0) for r in results) / len(results) if results else 0
        
        labels = [r.get("source_id", f"File {i}") for i, r in enumerate(results)]
        mahal_data = [r.get("mahalanobis_distance", 0) for r in results]
        conf_data = [r.get("confidence", 0) for r in results]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Smart-Code-Reviewer Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; margin: 20px; }}
        .card-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #1e1e1e; padding: 20px; border-radius: 8px; flex: 1; text-align: center; border: 1px solid #333; }}
        .card h3 {{ margin: 0; color: #888; font-size: 14px; }}
        .card p {{ font-size: 24px; font-weight: bold; margin: 10px 0 0 0; color: #00e5ff; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e1e1e; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #252525; color: #888; cursor: pointer; }}
        tr:hover {{ background: #2a2a2a; }}
        .anomalous {{ color: #ff5252; font-weight: bold; }}
        .chart-container {{ background: #1e1e1e; padding: 20px; border-radius: 8px; margin-bottom: 30px; border: 1px solid #333; }}
    </style>
</head>
<body>
    <h1>Smart-Code-Reviewer Analysis Report</h1>
    <div class="card-container">
        <div class="card"><h3>Total Files</h3><p>{len(results)}</p></div>
        <div class="card"><h3>Anomalies</h3><p>{anomalies}</p></div>
        <div class="card"><h3>Avg Confidence</h3><p>{avg_confidence:.4f}</p></div>
    </div>
    
    <div class="chart-container">
        <canvas id="mahalChart" height="100"></canvas>
    </div>

    <table>
        <thead>
            <tr>
                <th>Source ID</th>
                <th>Mahalanobis</th>
                <th>Entropy</th>
                <th>Z-Score</th>
                <th>Confidence</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {"".join(f'''
            <tr>
                <td>{r.get("source_id", "N/A")}</td>
                <td>{r.get("mahalanobis_distance", 0):.4f}</td>
                <td>{r.get("entropy", 0):.4f}</td>
                <td>{r.get("z_score", 0):.4f}</td>
                <td>{r.get("confidence", 0):.4f}</td>
                <td class="{"anomalous" if r.get("is_anomalous") else ""}">{ "ANOMALY" if r.get("is_anomalous") else "NORMAL"}</td>
            </tr>''' for r in results)}
        </tbody>
    </table>

    <script>
        const ctx = document.getElementById('mahalChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Mahalanobis Distance',
                    data: {json.dumps(mahal_data)},
                    backgroundColor: 'rgba(0, 229, 255, 0.5)',
                    borderColor: 'rgba(0, 229, 255, 1)',
                    borderWidth: 1
                }}, {{
                    label: 'Confidence',
                    data: {json.dumps(conf_data)},
                    type: 'line',
                    borderColor: '#ffeb3b',
                    fill: false
                }}]
            }},
            options: {{
                scales: {{ y: {{ beginAtZero: true, grid: {{ color: '#333' }} }} }}
            }}
        }});
    </script>
</body>
</html>
        """
        with open(path, "w") as f:
            f.write(html)
        return path
