"""
web_app.py
A clean web dashboard for the Multimodal Attendance System.
Connects directly to the existing SQLite database (db.py).
Provides endpoints for:
  - Attendance Dashboard (/)
  - Enrolled Students & Enrollment Form (/students)
  - Quick Attendance Marking (/mark-attendance)
  - CSV Export (/export)
  - REST API for load testing (/api/attendance)
"""

import os
import sys
import csv
import io
from datetime import datetime
from flask import Flask, render_template_string, request, redirect, url_for, Response, jsonify

# Add app directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db

app = Flask(__name__)
db.init_db()

BASE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - Multimodal Attendance System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f4f6f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .navbar-brand { font-weight: 700; letter-spacing: 0.5px; }
        .card { border: none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
        .stat-card { border-left: 5px solid #0d6efd; }
        .stat-card-success { border-left: 5px solid #198754; }
        .stat-card-info { border-left: 5px solid #0dcaf0; }
        .badge-present { background-color: #d1e7dd; color: #0f5132; }
        .badge-unverified { background-color: #fff3cd; color: #664d03; }
        .table-responsive { border-radius: 8px; overflow: hidden; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary shadow-sm mb-4">
        <div class="container">
            <a class="navbar-brand" href="/">🎓 Multimodal Attendance System</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link text-white px-3" id="nav-dashboard" href="/">Dashboard</a>
                <a class="nav-link text-white px-3" id="nav-students" href="/students">Students</a>
                <a class="nav-link text-white px-3" id="nav-mark" href="/mark-attendance">Mark Attendance</a>
                <a class="nav-link btn btn-light text-primary ms-2 px-3 fw-bold" id="nav-export" href="/export">Export CSV</a>
            </div>
        </div>
    </nav>

    <div class="container">
        {% block content %}{% endblock %}
    </div>

    <footer class="text-center text-muted py-4 mt-5">
        <small>Software Engineering Project &bull; Multimodal Attendance System (Face + Voice)</small>
    </footer>
</body>
</html>
"""

DASHBOARD_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
    <div class="row mb-4">
        <div class="col-md-4">
            <div class="card p-3 stat-card bg-white">
                <div class="text-muted small">TOTAL ENROLLED STUDENTS</div>
                <h2 class="fw-bold mb-0 text-primary" id="stat-total-students">{{ students|length }}</h2>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card p-3 stat-card-success bg-white">
                <div class="text-muted small">TOTAL ATTENDANCE RECORDS</div>
                <h2 class="fw-bold mb-0 text-success" id="stat-total-records">{{ records|length }}</h2>
            </div>
        </div>
        <div class="col-md-4">
            <div class="card p-3 stat-card-info bg-white">
                <div class="text-muted small">TODAY'S DATE</div>
                <h2 class="fw-bold mb-0 text-info" id="stat-date">{{ today }}</h2>
            </div>
        </div>
    </div>

    <div class="card p-4 bg-white mb-4">
        <div class="d-flex justify-content-between align-items-center mb-3">
            <h4 class="mb-0 fw-bold">📋 Attendance Logs</h4>
            <form method="GET" action="/" class="d-flex gap-2">
                <input type="date" name="date" class="form-control form-control-sm" value="{{ selected_date }}" id="filter-date">
                <button type="submit" class="btn btn-sm btn-outline-primary" id="btn-filter">Filter</button>
                {% if selected_date %}
                <a href="/" class="btn btn-sm btn-outline-secondary">Reset</a>
                {% endif %}
            </form>
        </div>

        <div class="table-responsive">
            <table class="table table-hover align-middle" id="attendance-table">
                <thead class="table-light">
                    <tr>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Roll No</th>
                        <th>Student Name</th>
                        <th>Status</th>
                        <th>Confidence (Face / Voice)</th>
                    </tr>
                </thead>
                <tbody>
                    {% for r in records %}
                    <tr>
                        <td>{{ r.date }}</td>
                        <td>{{ r.time }}</td>
                        <td><span class="badge bg-secondary">{{ r.roll_no }}</span></td>
                        <td class="fw-semibold">{{ r.name }}</td>
                        <td>
                            <span class="badge {{ 'badge-present' if r.status == 'Present' else 'badge-unverified' }} px-2 py-1">
                                {{ r.status }}
                            </span>
                        </td>
                        <td>
                            <small class="text-muted">
                                Face: {{ '%.1f'|format(r.face_confidence) if r.face_confidence is not none else 'N/A' }} | 
                                Voice: {{ '%.1f'|format(r.voice_confidence) if r.voice_confidence is not none else 'N/A' }}
                            </small>
                        </td>
                    </tr>
                    {% else %}
                    <tr>
                        <td colspan="6" class="text-center text-muted py-4">No attendance records found for this selection.</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
""")

STUDENTS_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
    <div class="row">
        <div class="col-md-5 mb-4">
            <div class="card p-4 bg-white">
                <h4 class="fw-bold mb-3">➕ Enroll Student</h4>
                {% if message %}
                <div class="alert alert-info py-2" id="alert-msg">{{ message }}</div>
                {% endif %}
                <form method="POST" action="/students" id="enroll-form">
                    <div class="mb-3">
                        <label class="form-label small fw-semibold">Student Full Name</label>
                        <input type="text" name="name" class="form-control" placeholder="e.g. Rahul Sharma" required id="input-name">
                    </div>
                    <div class="mb-3">
                        <label class="form-label small fw-semibold">Roll Number / Student ID</label>
                        <input type="text" name="roll_no" class="form-control" placeholder="e.g. 24BAI1560" required id="input-roll">
                    </div>
                    <button type="submit" class="btn btn-primary w-100 fw-semibold" id="btn-enroll-submit">Enroll Student</button>
                </form>
            </div>
        </div>

        <div class="col-md-7 mb-4">
            <div class="card p-4 bg-white">
                <h4 class="fw-bold mb-3">👥 Enrolled Students List</h4>
                <div class="table-responsive">
                    <table class="table table-hover align-middle" id="students-table">
                        <thead class="table-light">
                            <tr>
                                <th>#</th>
                                <th>Name</th>
                                <th>Roll No</th>
                                <th>Enrolled At</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for s in students %}
                            <tr>
                                <td>{{ s.id }}</td>
                                <td class="fw-bold text-primary">{{ s.name }}</td>
                                <td><span class="badge bg-dark">{{ s.roll_no }}</span></td>
                                <td><small class="text-muted">{{ s.enrolled_at }}</small></td>
                            </tr>
                            {% else %}
                            <tr>
                                <td colspan="4" class="text-center text-muted py-3">No students enrolled yet.</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
""")

MARK_HTML = BASE_HTML.replace("{% block content %}{% endblock %}", """
    <div class="row justify-content-center">
        <div class="col-md-6">
            <div class="card p-4 bg-white">
                <h4 class="fw-bold mb-3">✍️ Mark Attendance (Manual / Verification)</h4>
                {% if message %}
                <div class="alert alert-{{ 'success' if status == 'ok' else 'warning' }} py-2" id="alert-mark-msg">{{ message }}</div>
                {% endif %}
                <form method="POST" action="/mark-attendance" id="mark-form">
                    <div class="mb-3">
                        <label class="form-label small fw-semibold">Select Student</label>
                        <select name="student_id" class="form-select" required id="select-student">
                            {% for s in students %}
                            <option value="{{ s.id }}">{{ s.name }} ({{ s.roll_no }})</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="mb-3">
                        <label class="form-label small fw-semibold">Verification Status</label>
                        <select name="status" class="form-select" id="select-status">
                            <option value="Present">Present (Multimodal Verified)</option>
                            <option value="Unverified">Unverified</option>
                        </select>
                    </div>
                    <button type="submit" class="btn btn-success w-100 fw-semibold" id="btn-mark-submit">Submit Attendance</button>
                </form>
            </div>
        </div>
    </div>
""")


@app.route("/")
def dashboard():
    selected_date = request.args.get("date", "").strip()
    records = db.list_attendance(selected_date or None)
    students = db.list_students()
    today = datetime.now().strftime("%Y-%m-%d")
    return render_template_string(
        DASHBOARD_HTML,
        title="Dashboard",
        records=records,
        students=students,
        today=today,
        selected_date=selected_date,
    )


@app.route("/students", methods=["GET", "POST"])
def students_page():
    message = None
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        if name and roll_no:
            try:
                db.add_student(name, roll_no)
                message = f"Successfully enrolled student {name} ({roll_no})!"
            except Exception as e:
                message = f"Error: Student roll number may already exist ({e})"
    students = db.list_students()
    return render_template_string(
        STUDENTS_HTML,
        title="Students",
        students=students,
        message=message,
    )


@app.route("/mark-attendance", methods=["GET", "POST"])
def mark_attendance_page():
    message = None
    status = None
    if request.method == "POST":
        student_id = request.form.get("student_id")
        att_status = request.form.get("status", "Present")
        if student_id:
            rec_id, res = db.record_attendance(
                student_id=int(student_id),
                face_confidence=95.5,
                voice_confidence=88.2,
                status=att_status,
            )
            status = res
            if res == "ok":
                message = "Attendance marked successfully!"
            else:
                message = "Attendance already marked for today."
    students = db.list_students()
    return render_template_string(
        MARK_HTML,
        title="Mark Attendance",
        students=students,
        message=message,
        status=status,
    )


@app.route("/export")
def export_csv():
    rows = db.list_attendance()
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=["date", "time", "name", "roll_no", "status", "face_confidence", "voice_confidence"]
    )
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r[k] for k in writer.fieldnames})
    
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=attendance_report.csv"},
    )


@app.route("/api/attendance")
def api_attendance():
    """REST endpoint for JMeter non-functional load testing"""
    records = db.list_attendance()
    return jsonify({
        "status": "success",
        "total_records": len(records),
        "records": records,
    })


if __name__ == "__main__":
    print("Starting Multimodal Attendance Web Dashboard on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
