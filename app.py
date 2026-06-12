import streamlit as st
import face_recognition
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

# -----------------------------------
# BASIC CONFIG
# -----------------------------------
st.set_page_config(page_title="Smart Attendance System", layout="centered")

IMAGES_FOLDER = "images"
STUDENTS_FILE = "students.csv"
ATTENDANCE_FILE = "Attendance.csv"
LOCATION_FILE = "location.txt"


# -----------------------------------
# INITIAL SETUP
# -----------------------------------
def setup_files():
    if not os.path.exists(IMAGES_FOLDER):
        os.makedirs(IMAGES_FOLDER)

    if not os.path.exists(STUDENTS_FILE):
        df = pd.DataFrame([
            ["anshu", "1234", "Anshu Kumar"],
            ["rahul", "5678", "Rahul Yadav"],
            ["priya", "abcd", "Priya Singh"],
            ["amit", "9999", "Amit Kumar"],
            ["neha", "2468", "Neha Sharma"]
        ], columns=["Username", "Password", "Name"])
        df.to_csv(STUDENTS_FILE, index=False)

    if not os.path.exists(ATTENDANCE_FILE):
        pd.DataFrame(columns=["Name", "Time", "Date", "Location"]).to_csv(ATTENDANCE_FILE, index=False)

    if not os.path.exists(LOCATION_FILE):
        with open(LOCATION_FILE, "w") as f:
            f.write("Not Set")


setup_files()


# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------
def load_students():
    return pd.read_csv(STUDENTS_FILE)


def get_location():
    with open(LOCATION_FILE, "r") as f:
        return f.read().strip()


def set_location(loc):
    with open(LOCATION_FILE, "w") as f:
        f.write(loc)


def load_encodings():
    encodings, names = [], []
    for file in os.listdir(IMAGES_FOLDER):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(f"{IMAGES_FOLDER}/{file}")
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_encodings(rgb)
            if faces:
                encodings.append(faces[0])
                names.append(os.path.splitext(file)[0])
    return encodings, names


def mark_attendance(name, location):
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%d-%m-%Y")

    if not ((df["Name"] == name) & (df["Date"] == today)).any():
        now = datetime.now().strftime("%H:%M:%S")
        new_row = {"Name": name, "Time": now, "Date": today, "Location": location}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return True
    return False


# -----------------------------------
# SESSION CONTROLS
# -----------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.name = None
    st.session_state.run_camera = False

st.title("🧠 Smart Attendance System")


# ============================================================
#                    LOGIN PAGE
# ============================================================
if not st.session_state.logged_in:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Student Login")
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")

        if st.button("Login as Student"):
            df = load_students()
            match = df[(df["Username"] == user) & (df["Password"] == pw)]

            if not match.empty:
                st.session_state.logged_in = True
                st.session_state.role = "student"
                st.session_state.name = match.iloc[0]["Name"]
                st.success(f"Welcome {st.session_state.name}")
            else:
                st.error("Invalid credentials ❌")

    with col2:
        st.subheader("Admin Login")
        auser = st.text_input("Admin Username")
        apw = st.text_input("Admin Password", type="password")

        if st.button("Login as Admin"):
            if auser == "admin" and apw == "admin123":
                st.session_state.logged_in = True
                st.session_state.role = "admin"
                st.session_state.name = "Administrator"
                st.success("Admin login successful")
            else:
                st.error("Invalid admin credentials ❌")


# ============================================================
#                    STUDENT DASHBOARD
# ============================================================
if st.session_state.logged_in and st.session_state.role == "student":

    st.subheader(f"👋 Welcome {st.session_state.name}")
    st.info(f"📍 Current Location: **{get_location()}**")

    st.markdown("---")
    st.subheader("🎥 Mark Attendance")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🟢 Start Camera"):
            st.session_state.run_camera = True

    with col2:
        if st.button("🔴 Stop Camera"):
            st.session_state.run_camera = False

    if st.session_state.run_camera:
        encodings, names = load_encodings()
        if len(encodings) == 0:
            st.error("⚠ No face data found in images folder.")
        else:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            found = False

            while st.session_state.run_camera:
                ok, frame = cap.read()
                if not ok:
                    st.error("❌ Camera not working")
                    break

                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                faces = face_recognition.face_locations(rgb_small)
                encs = face_recognition.face_encodings(rgb_small, faces)

                for enc, loc in zip(encs, faces):
                    matches = face_recognition.compare_faces(encodings, enc)
                    face_dist = face_recognition.face_distance(encodings, enc)
                    best = np.argmin(face_dist)

                    if matches[best]:
                        name = names[best]
                        y1, x2, y2, x1 = [v * 4 for v in loc]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        cv2.putText(frame, "Attendance Marked!", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                        stframe.image(frame, channels="BGR", width="stretch")
                        time.sleep(1.5)

                        if mark_attendance(name, get_location()):
                            st.success(f"Attendance marked for {name} ✔️")
                        else:
                            st.info(f"{name} already marked today")

                        st.session_state.run_camera = False
                        found = True
                        break

                stframe.image(frame, channels="BGR", width="stretch")

            cap.release()
            st.info("Camera stopped")

    st.markdown("---")
    st.subheader("📋 My Attendance Record")

    df = pd.read_csv(ATTENDANCE_FILE)
    my = df[df["Name"] == st.session_state.name]

    if len(my) > 0:
        st.dataframe(my.sort_values(by=["Date", "Time"], ascending=False), width="stretch")

        st.download_button(
            "📥 Download My Attendance",
            my.to_csv(index=False).encode(),
            file_name=f"{st.session_state.name}_attendance.csv",
            mime="text/csv"
        )
    else:
        st.info("No records found.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()


# ============================================================
#                    ADMIN DASHBOARD
# ============================================================
if st.session_state.logged_in and st.session_state.role == "admin":

    st.subheader("👨‍💼 Admin Dashboard")

    loc = st.text_input("Set Location", value=get_location())
    if st.button("Save Location"):
        set_location(loc)
        st.success("Location updated ✔")

    st.markdown("---")
    st.subheader("📊 Attendance Records")

    df = pd.read_csv(ATTENDANCE_FILE)

    if len(df) > 0:
        options = ["All"] + sorted(df["Name"].unique())
        selected = st.selectbox("Filter by Name", options)

        if selected != "All":
            df = df[df["Name"] == selected]

        st.dataframe(df, width="stretch")

        st.download_button(
            "📥 Download Full CSV",
            df.to_csv(index=False).encode(),
            file_name="Attendance_Report.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records exist.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

