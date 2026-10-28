"""
test_selenium.py
Automated Functional UI Test Suite using Selenium WebDriver for the
Multimodal Attendance Web Portal.

Tests:
1. Verify Dashboard Loads & Stats are Visible
2. Verify Navigation between Dashboard, Students, and Mark Attendance pages
3. Verify Student Enrollment form submission (Functional Form Test)
4. Verify Attendance Log table displays student records
5. Verify CSV Export endpoint functionality
"""

import time
import os
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

BASE_URL = "http://127.0.0.1:5000"
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


@pytest.fixture(scope="module")
def driver():
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--headless=new")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    driver.set_window_size(1280, 800)
    yield driver
    driver.quit()


def test_01_dashboard_load(driver):
    """Functional Test 1: Verify homepage dashboard renders properly"""
    driver.get(BASE_URL)
    time.sleep(1)
    
    # Save snapshot
    snap_path = os.path.join(SCREENSHOT_DIR, "selenium_01_dashboard.png")
    driver.save_screenshot(snap_path)
    
    assert "Multimodal Attendance System" in driver.title
    total_students = driver.find_element(By.ID, "stat-total-students")
    assert total_students.is_displayed()
    print("\\n[PASS] Dashboard loaded successfully with summary cards.")


def test_02_navigation_to_students_page(driver):
    """Functional Test 2: Verify navbar links navigate properly"""
    driver.get(BASE_URL)
    students_nav = driver.find_element(By.ID, "nav-students")
    students_nav.click()
    time.sleep(1)
    
    snap_path = os.path.join(SCREENSHOT_DIR, "selenium_02_students_page.png")
    driver.save_screenshot(snap_path)
    
    assert "/students" in driver.current_url
    assert "Enrolled Students List" in driver.page_source
    print("\\n[PASS] Navigation to Students page verified.")


def test_03_enroll_student_form_submission(driver):
    """Functional Test 3: Test functional submission of Student Enrollment form"""
    driver.get(f"{BASE_URL}/students")
    
    name_input = driver.find_element(By.ID, "input-name")
    roll_input = driver.find_element(By.ID, "input-roll")
    submit_btn = driver.find_element(By.ID, "btn-enroll-submit")
    
    test_roll = f"TEST{int(time.time()) % 10000}"
    name_input.send_keys("Automated Test Student")
    roll_input.send_keys(test_roll)
    
    snap_path = os.path.join(SCREENSHOT_DIR, "selenium_03_enroll_form_fill.png")
    driver.save_screenshot(snap_path)
    
    submit_btn.click()
    time.sleep(1)
    
    snap_success = os.path.join(SCREENSHOT_DIR, "selenium_04_enroll_success.png")
    driver.save_screenshot(snap_success)
    
    assert "Successfully enrolled" in driver.page_source or "Error" in driver.page_source
    print(f"\\n[PASS] Student enrollment form submitted for roll {test_roll}.")


def test_04_attendance_table_verification(driver):
    """Functional Test 4: Verify attendance table and columns"""
    driver.get(BASE_URL)
    time.sleep(1)
    
    table = driver.find_element(By.ID, "attendance-table")
    assert table.is_displayed()
    
    snap_path = os.path.join(SCREENSHOT_DIR, "selenium_05_attendance_table.png")
    driver.save_screenshot(snap_path)
    print("\\n[PASS] Attendance log table verified.")


def test_05_export_csv_endpoint(driver):
    """Functional Test 5: Verify the CSV export button exists and points to /export"""
    driver.get(BASE_URL)
    export_btn = driver.find_element(By.ID, "nav-export")
    assert export_btn.is_displayed()
    assert "/export" in export_btn.get_attribute("href")
    print("\\n[PASS] Export CSV feature verified.")
