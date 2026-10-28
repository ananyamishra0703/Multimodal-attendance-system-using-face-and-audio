"""
main.py
Entry point and CLI menu for the Multimodal (Face + Voice) Attendance
System. Run from the project root with:

    python3 app/main.py
"""

import os
import csv
import sys

import db
import enrollment
import attendance


def view_attendance_log():
    print("\n--- Attendance Log ---")
    date_filter = input("Filter by date (YYYY-MM-DD), or press Enter for all: ").strip()
    rows = db.list_attendance(date_filter or None)

    if not rows:
        print("No attendance records found.\n")
        return

    print(f"\n{'Date':<12} {'Time':<10} {'Name':<22} {'Roll No':<10} {'Status':<12}")
    print("-" * 68)
    for r in rows:
        print(f"{r['date']:<12} {r['time']:<10} {r['name']:<22} {r['roll_no']:<10} {r['status']:<12}")
    print()


def export_attendance_csv():
    print("\n--- Export Attendance to CSV ---")
    date_filter = input("Filter by date (YYYY-MM-DD), or press Enter for all: ").strip()
    rows = db.list_attendance(date_filter or None)

    if not rows:
        print("No attendance records to export.\n")
        return

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "exports")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"attendance_{date_filter or 'all'}.csv"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "time", "name", "roll_no", "status", "face_confidence", "voice_confidence"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print(f"Exported {len(rows)} record(s) to {out_path}\n")


def list_enrolled_students():
    print("\n--- Enrolled Students ---")
    students = db.list_students()
    if not students:
        print("No students enrolled yet.\n")
        return
    print(f"\n{'ID':<5} {'Name':<25} {'Roll No':<12} {'Enrolled At'}")
    print("-" * 60)
    for s in students:
        print(f"{s['id']:<5} {s['name']:<25} {s['roll_no']:<12} {s['enrolled_at']}")
    print()


MENU = """
=========================================
 Multimodal Attendance System (Face + Voice)
=========================================
1. Enroll new student
2. Take attendance
3. View attendance log
4. Export attendance to CSV
5. List enrolled students
6. Exit
"""


def main():
    db.init_db()
    while True:
        print(MENU)
        choice = input("Choose an option (1-6): ").strip()

        if choice == "1":
            enrollment.enroll_new_student()
        elif choice == "2":
            attendance.run_attendance_session()
        elif choice == "3":
            view_attendance_log()
        elif choice == "4":
            export_attendance_csv()
        elif choice == "5":
            list_enrolled_students()
        elif choice == "6":
            print("Goodbye.")
            sys.exit(0)
        else:
            print("Invalid option, please choose 1-6.\n")


if __name__ == "__main__":
    main()
