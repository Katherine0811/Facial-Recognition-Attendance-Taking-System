import smtplib
from smtplib import SMTPException
import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
import sqlite3

import cv2
import os
import numpy as np
import pickle
import random
import time
import fnmatch

# Main Frame of the System: root
root = Tk()
root.title('Face Recognition Attendance Taking System')
root.geometry('1000x500')
root.configure(background = '#FAEEE0')

# root.iconbitmap('')

# Initialize Database Function
def database():
    # Database for Student
    con = sqlite3.connect('studentdata.db')
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS record(
                        name text,
                        studentID integer PRIMARY KEY, 
                        contact number, 
                        gender text, 
                        email text, 
                        password text
                    )
                ''')
    con.commit()

    # Database for Staff
    con = sqlite3.connect('staffdata.db')
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS record(
                        name text,
                        staffID integer PRIMARY KEY, 
                        contact number, 
                        gender text, 
                        email text, 
                        password text
                    )
                ''')
    con.commit()

    # Database for Admin
    con = sqlite3.connect('admindata.db')
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS record(
                        name text,
                        adminID integer PRIMARY KEY, 
                        contact number, 
                        gender text, 
                        email text, 
                        password text
                    )
                ''')
    con.commit()

    # Database for Attendance
    con = sqlite3.connect('attendance.db')
    cur = con.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS record(
                        studentName text,
                        studentID integer PRIMARY KEY, 
                        studentEmail text, 
                        digitCode integer,
                        attendance bool
                    )
                ''')
    con.commit()


# Train Facial Recognition Model
def facesTrain():
    start = time.time()
    print("Database is undergoing training")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get Directory File Location
    image_dir = os.path.join(BASE_DIR, "Images")

    face_cascade = cv2.CascadeClassifier(
        'FYP2/Cascades/data/haarcascade_frontalface_default.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    student_name = ""  # Name
    label_ids = {}  # Dictionary
    x_train = []  # Numbers of Pixel Value
    y_labels = []  # Numbers Related to Lable
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                path = os.path.join(root, file)
                
                # All Space Bar is Replaced With Dash And Names are in Lower Capital (Standardize)
                id_ = int(os.path.basename(os.path.dirname(path)))
                # print(id_, path)
                # EXP: 2005661 c:\Users\kathe\Desktop\FYP\LeeYanJie\FYP2\Images\2005661\Lee_Yan_Jie_(1).jpg

                if not id_ in label_ids:
                    con = sqlite3.connect('attendance.db')
                    cur = con.cursor()
                    cur.execute("SELECT * FROM record")
                    records = cur.fetchall()

                    for record in records:
                        if id_ == record[1]:
                            student_name = record[0].replace(" ", "-").lower()

                    label_ids[id_] = student_name

                label = label_ids[id_] # 'id_' is the Array Value of 'label_ids'
                # print(label)
                # EXP: lee-yan-jie

                # print(label_ids)
                # EXP: {'2005654': 'ameesha-jeet', '2005655': 'chong-en-han', '2005656': 'foo-jia-qi', '2005657': 'heng-zhi-xuan', '2005658': 'lee-quan-jin', '2005659': 'lee-xin-wei', '2005660': 'lee-xin-yun', '2005661': 'lee-yan-jie'}


                # Image Processing
                pil_image = Image.open(path).convert("L")  # Grayscale
                size = (550, 550) # Scale Image Size
                final_image = pil_image.resize(size, Image.ANTIALIAS)

                image_array = np.array(final_image, "uint8")
                # print(image_array)
                # Converts Image into Numbers (Metrix ?)


                faces = face_cascade.detectMultiScale(
                    image_array, scaleFactor = 1.05, minNeighbors = 6)
                
                # ScaleFactor 
                # Specifies how much the image size is reduced at each image scale.
                # 1.05 increase chances of matching size but algorithm works slower since it is more thorough.
                # 1.40 means faster detection but risk of missing faces altogether.
                # Suggest 1.05 - 1.40
                # Previous testing at 1.5

                # Neighbours
                # Depict the number of sample points that is used to build the circular local binary pattern.
                # More number of points, higher computational cost, less detections but higher quality.
                # Suggest 3 - 6
                # Originally set to 8, previous testing at 5

                for (x, y, w, h) in faces:
                    roi = image_array[y: y + h, x: x + w]
                    x_train.append(roi) # Verify Image, turn into a NUMPY array, colored gray
                    y_labels.append(id_)  # Add Number

    con.commit()
    con.close()

    with open("labels.pickle", 'wb') as f:  # writing byter
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainner.yml")

    print("Database has been successfully trained")
    end = time.time()
    total = end - start
    print(total)


# Use Facial Recognition Model
def faces(user_name, user_id):
    # Face Detection - Haar Cascade
    face_cascade = cv2.CascadeClassifier('FYP2/cascades/data/haarcascade_frontalface_default.xml')

    # Train Model - LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")

    # What do you do?
    name = ""
    total = 0
    verify_atd = False
    labels = {1: "person_name"}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {k: v for k, v in og_labels.items()}
        
        # print(labels)
        # EXP: {2005654: 'ameesha-jeet', 2005655: 'chong-en-han', 2005656: 'foo-jia-qi', 2005657: 'heng-zhi-xuan', 2005658: 'lee-quan-jin', 2005659: 'lee-xin-wei', 2005660: 'lee-xin-yun', 2005661: 'lee-yan-jie', 2005662: 'lee-ying-shan', 2005663: 'lim-shin-yie', 2005664: 'mok-yuen-yua', 2005665: 'seah-wei-ming', 2005666: 'siew-wai-han', 2005667: 'tan-choy-yin', 2005668: 'tan-yoke-shuen', 2005669: 'tan-zhe-yan', 2005670: 'tia-wan-tong', 2005671: 'wong-kar-lok', 2005672: 'yap-wei-xiang', 2005673: 'yap-yong-yi'}

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    start = time.time()

    # Open Camera
    while True:
        # Capture Frame-by-Frame
        ret, frame = cap.read()
        # Haar Cascades Works in Gray Image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 6, minSize = (200, 200))

        for (x, y, w, h) in faces:
            # (x, y, w, h) Detect Faces Region
            # It is the Coordinates Where w = Width and h = Height
            roi_gray = gray[y: y + h, x: x + w]
            # [] = Location of This Frame
            roi_color = frame[y: y + h, x: x + w]

            # Recognizer (Deep Learned Model Predict)
            id_, conf = recognizer.predict(roi_gray)

            if 10 < conf < 50:  # confidence level 0 = exact match
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2

                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
                # cv2.putText(frame, str(conf), (x + w, y + h), font, 1, color, stroke, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

            end = time.time()
            total = end - start
            
            if (name == (user_name.get().replace(" ", "-").lower()) and str(id_) == user_id.get()):
                verify_atd = True
                break

        # Display the resulting frame (color)
        cv2.imshow('frame', frame)

        keyCode = cv2.waitKey(1)
        if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
            break

        elif verify_atd == True:
            takeAttendance(user_name, user_id)
            print(total)
            break

        if ((total) > 10): 
            messagebox.showerror('Error', 'User Not Defined')
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


# User Verification Panel - Student (for FR Attendance)
def confirmIdentity():
    root.withdraw()
    fr_atd = Toplevel()
    fr_atd.title('Face Recognition Portal')
    fr_atd.geometry('440x270')
    fr_atd.configure(background = '#DBD0C0')

    # Get User Name, ID and Pin Code
    title_lbl = Label(fr_atd, text = "  Verify Student Identity  ",
                    bg = '#DBD0C0', font = ('helvetica', 15, 'bold underline'))
    title_lbl.place(x = 220, y = 30, anchor = "n")

    user_name_lbl = Label(fr_atd, text = "Student Name",
                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
    user_name_lbl.place(x = 50, y = 100)

    user_id_lbl = Label(fr_atd, text = "Student ID",
                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
    user_id_lbl.place(x = 50, y = 125)

    user_atd_code_lbl = Label(fr_atd, text = "Attendance Code",
                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
    user_atd_code_lbl.place(x = 50, y = 150)

    user_name = Entry(fr_atd, width = 30)
    user_name.place(x = 200, y = 100)

    user_id = Entry(fr_atd, width = 30)
    user_id.place(x = 200, y = 125)

    user_atd_code = Entry(fr_atd, width = 30)
    user_atd_code.place(x = 200, y = 150)

    # Create Change Password Button
    btn_take_atd = Button(fr_atd, text = "Scan Face for Attendance",
                            command = lambda: checkIdentity(user_name, user_id, user_atd_code),
                            bg = '#FAEEE0', width = 32, height = 1, borderwidth = 3,
                            relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
    btn_take_atd.place(x = 220, y = 200, anchor = "n")  
    
    # Exit Protocol
    fr_atd.wm_protocol("WM_DELETE_WINDOW", lambda: exit(fr_atd, root))


# User Verification Function - Student (for FR Attendance)
def checkIdentity(user_name, user_id, user_atd_code):
    check = False # Check For Empty Input Field
    warn = "" # Error Message Will Be Specificly Displayed

    try:
        con = sqlite3.connect('attendance.db')
        cur = con.cursor()
        
        for record in cur.execute("SELECT * FROM record"):

            if (user_name.get() == "" or user_id.get() == "" or user_atd_code.get() == ""):
                warn = "Please don't Leave Blank!"
            elif (record[0] == user_name.get() and str(record[1]) == str(user_id.get()) and str(record[3]) == str(user_atd_code.get())):
                messagebox.showinfo(
                    'Validate Status', 'Student Identified Successfully!')
                check = True
            else:
                warn = "Invalid user!"

    except Exception as ep:
        messagebox.showerror('', ep)

    con.commit()
    con.close()

    if check == False:
        messagebox.showerror('Error', warn)
    else:
        faces(user_name, user_id)


# User Verification Function (for Forget & Reset Password)
def verifyIdentity(number, user_name, user_id, user_email, digit_code):
    check = False # Check For Empty Input Field
    warn = "" # Error Message Will Be Specificly Displayed
    user_type = ""

    try:
        if number == 1:
            con = sqlite3.connect('studentdata.db')
            user_type = "Student"
        elif number == 2:
            con = sqlite3.connect('staffdata.db')
            user_type = "Staff"
            
        cur = con.cursor()
        
        for record in cur.execute("SELECT * FROM record"):

            if (user_name.get() == "" or user_id.get() == "" or user_email.get() == ""):
                warn = "Please don't Leave Blank!"
                
            elif (record[0] == user_name.get() and str(record[1]) == str(user_id.get()) and str(record[4]) == str(user_email.get())):
                check = True
                sender_email = "katherine.lee0356@gmail.com"
                sender_password = "wsfpoihmjtogkdnz"
                receiver_email = str(user_email.get())
                subject = "Digit Code to Reset Your TARUC " + user_type + " Password"
                main_message = "Your Digit Code is <" + str(digit_code) + ">. Please Verify " + user_type + " Identity to Set a New Password. "

                Body = """
                    From: Katherine <%s>
                    To: <%s>
                    Subject: %s 

                    %s
                """ %(sender_email, receiver_email, subject, main_message)

                try:
                    server = smtplib.SMTP('smtp.gmail.com', 587)
                    server.starttls()
                    server.login(sender_email, sender_password)
                    server.sendmail(sender_email, receiver_email, Body)

                    messagebox.showinfo(
                    'Validate Status', 'Identity Verified Successfully! \n\nA 6 Digit Code has been sent to <' + user_email.get() + '> !')

                except (smtplib.SMTPException, ConnectionRefusedError, OSError):
                    return

                finally:
                    server.quit()
            else:
                warn = "Invalid user!"

    except Exception as ep:
        messagebox.showerror('', ep)

    con.commit()
    con.close()

    if check == False:
        messagebox.showerror('Error', warn)


# Confirm Reset Password Function
def forgetPassword(panel, user_type, number, user_id, digit_code, psw_change_code):
    status = "change_psw_2"
    check = False # Check For Empty Input Field
    warn = "" # Error Message Will Be Specificly Displayed

    try:
        if str(psw_change_code.get()) == "":
            warn = "Please don't Leave Blank!"
        elif str(psw_change_code.get()) != str(digit_code):
            warn = "Invalid Digit Code!"
        else:
            check = True
            print("Verified")
            id = user_id.get()
            updateRecord(panel, status, user_type, number, id)

    except Exception as ep:
        messagebox.showerror('', ep)

    if check == False:
        messagebox.showerror('Error', warn)


# Generate User ID Function
def generateID(number):
    if number == 1:
        con = sqlite3.connect('studentdata.db')
        cur = con.cursor()
        cur.execute("SELECT * FROM record ORDER BY StudentID DESC LIMIT 1")
    elif number == 2:
        con = sqlite3.connect('staffdata.db')
        cur = con.cursor()
        cur.execute("SELECT * FROM record ORDER BY StaffID DESC LIMIT 1")
    elif number == 3:
        con = sqlite3.connect('admindata.db')
        cur = con.cursor()
        cur.execute("SELECT * FROM record ORDER BY AdminID DESC LIMIT 1")
    
    records = cur.fetchone()
    count = 0
    for record in records:
        count += 1
        if count == 2:
            user_id = record + 1
    
    return user_id


# Generate Digit Code Function
def generateCode(user_name, user_id, user_email, pin, attendance):
    try:
        while True:
            con = sqlite3.connect('attendance.db')
            cur = con.cursor()
            cur.execute("SELECT * FROM record")
            records = cur.fetchall()
            pin = random.randint(100000, 999999)
            for record in records:
                if pin == record[3]:
                    continue
            break

        cur.execute(
            "INSERT INTO record VALUES (:studentName, :studentID, :studentEmail, :digitCode, :attendance)",
            {
                'studentName': user_name.get(),
                'studentID': user_id.get(),
                'studentEmail': user_email.get(),
                'digitCode': pin,
                'attendance': attendance
            }
        )

        con.commit()
        con.close()

    except Exception as ep:
        messagebox.showerror('', ep)


# Login Panel
def homeLogin(number):
    root.withdraw()
    login = Toplevel()
    login.title('Login Portal')
    login.geometry('1000x500')
    login.configure(background = '#DBD0C0')

    if number == 1:
        user_type = "student"
    elif number == 2:
        user_type = "staff"
    elif number == 3:
        user_type = "admin"
    else:
        messagebox.showerror('Error', 'User Not Defined')

    # Create Textbox Labels
    login_lbl = Label(login, text = "  Please Enter Your Information  ",
                      bg = '#DBD0C0', font = ('helvetica', 15, 'bold underline'))
    login_lbl.place(x = 300, y = 125)

    email_lbl = Label(login, text = "Enter Email",
                      bg = '#DBD0C0', font = ('times', 15, 'bold'))
    email_lbl.place(x = 300, y = 200)

    psw_lbl = Label(login, text = "Enter Password",
                    bg = '#DBD0C0', font = ('times', 15, 'bold'))
    psw_lbl.place(x = 300, y = 250)

    # Create Textbox
    email_txt = Entry(login, width = 40, font = ('times', 15, 'bold'))
    email_txt.place(x = 450, y = 200)
    
    email_txt.insert(0, "@" + user_type + ".tarc.edu.my")

    psw_txt = Entry(login, width = 40, font = ('times', 15, 'bold'), show = '*')
    psw_txt.place(x = 450, y = 250)

    # Create Login Button
    login_btn = Button(login, text = "Login",
                       command = lambda: checkLogin(number, email_txt, psw_txt, login),
                       bg = '#FAEEE0', width = 20, height = 2, borderwidth = 3,
                       relief = "ridge", activebackground = '#F9E4C8', font = ('times', 15, 'bold'))
    login_btn.place(x = 300, y = 300)

    # Create Forget Button
    status = "forget_psw"
    id = ""
    forget_btn = Button(login, text = "Forgotten Your Password ?",
                        command = lambda: updateRecord(login, status, user_type, number, id),
                        bg = '#FAEEE0', width = 20, height = 2, borderwidth = 3,
                        relief = "ridge", activebackground = '#F9E4C8', font = ('times', 15, 'bold'))
    forget_btn.place(x = 600, y = 300)

    # Exit Protocol
    login.wm_protocol("WM_DELETE_WINDOW", lambda: exit(login, root))


# Check Login Validity Function
def checkLogin(number, email_txt, psw_txt, login):
    check = False  # Check For Empty Input Field
    warn = ""  # Error Message Will Be Specificly Displayed
    id = ""
    email_txt = email_txt.get()
    psw_txt = psw_txt.get()
    if email_txt == "":
        warn = "Email Address can't be empty!"
    elif psw_txt == "":
        warn = "Password can't be empty!"
    else:
        try:  # Select Which Database to Validate Email and Password
            if number == 1:
                con = sqlite3.connect('studentdata.db')
            elif number == 2:
                con = sqlite3.connect('staffdata.db')
            elif number == 3:
                con = sqlite3.connect('admindata.db')

            cur = con.cursor()
            cur.execute("SELECT * FROM record")
            records = cur.fetchall()

            for record in records:
                if email_txt == record[4]:
                    id = record[1]
                    if psw_txt == record[5]:
                        messagebox.showinfo(
                            'Login Status', 'Logged in Successfully!')
                        login.withdraw()
                        if number == 1:
                            profile(login, number, id)
                        elif number == 2:
                            profile(login, number, id)
                        elif number == 3:
                            admin(login, id)
                        
                        check = True
                        break

                warn = "Invalid email address or password!"

        except Exception as ep:
            messagebox.showerror('', ep)

        con.commit()
        con.close()

    if check == False:
        messagebox.showerror('', warn)


# Profile Panel
def profile(panel, number, id):
    profile = Toplevel()
    profile.title('User Profile Portal')
    if number == 1:
        profile.geometry('500x600')
    else:
        profile.geometry('500x350')
    profile.configure(background = '#FAEEE0')

    global img2

    try:  # Select Which Database to Validate Email and Password
        if number == 1:
            con = sqlite3.connect('studentdata.db')
            user_type = "Student"
            space = 200
        elif number == 2:
            con = sqlite3.connect('staffdata.db')
            user_type = "Staff"
            space = 0
        elif number == 3:
            con = sqlite3.connect('admindata.db')
            user_type = "Admin"
            space = 0
        else:
            messagebox.showerror('Error', 'User Not Defined')

        cur = con.cursor()
        cur.execute("SELECT * FROM record")
        records = cur.fetchall()

        # Get User Information
        for record in records:
            if str(id) == str(record[1]):
                user_name = str(record[0])
                user_id = str(record[1])
                user_contact = str(record[2])
                user_gender = str(record[3])
                user_email = str(record[4])
                user_password = str(record[5])

    except Exception as ep:
        messagebox.showerror('', ep)

    con.commit()
    con.close()

    user_password = '*' * len(user_password) # Hide Password into Asterisk

    # Create Text Box Labels
    title_lbl = Label(profile, text = "  " + user_type + " Profile  ",
                      bg = '#FAEEE0', font = ('helvetica', 15, 'bold underline'))
    title_lbl.place(x = 250, y = 30, anchor = "n")

    user_name_lbl = Label(profile, text = user_type + " Name", # Profile: User Name
                          bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_name_lbl.place(x = 50, y = space + 75)

    user_name_txt = Label(profile, text = user_name,
                          bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_name_txt.place(x = 200, y = space + 75)

    user_ID_lbl = Label(profile, text = user_type + " ID", # Profile: User ID
                        bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_ID_lbl.place(x = 50, y = space + 100)

    user_ID_txt = Label(profile, text = user_id,
                        bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_ID_txt.place(x = 200, y = space + 100)

    user_contact_lbl = Label(profile, text = "Contact Number", # Profile: User Contact No.
                             bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_contact_lbl.place(x = 50, y = space + 125)

    user_contact_txt = Label(profile, text = user_contact,
                             bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_contact_txt.place(x = 200, y = space + 125)

    user_gender_lbl = Label(profile, text = "Gender", # Profile: User Gender
                            bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_gender_lbl.place(x = 50, y = space + 150)

    user_gender_txt = Label(profile, text = user_gender,
                            bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_gender_txt.place(x = 200, y = space + 150)

    user_email_lbl = Label(profile, text = "Email Address", # Profile: User Email Address
                           bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_email_lbl.place(x = 50, y = space + 175)

    user_email_txt = Label(profile, text = user_email,
                           bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_email_txt.place(x = 200, y = space + 175)

    user_password_lbl = Label(profile, text = "Password", # Profile: User Password
                              bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_password_lbl.place(x = 50, y = space + 200)

    user_password_txt = Label(profile, text = user_password,
                              bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
    user_password_txt.place(x = 200, y = space + 200)

    if number == 1:
        # Get User Profile Pic
        directory = r'C:\\Users\\kathe\Desktop\\FYP\\LeeYanJie\\FYP2\\Images\\' + user_id
        os.chdir(directory)

        img_item = (user_name.replace(" ", "_") + "_(1).jpg")
        
        if os.path.isfile(img_item):
            img2 = Image.open(img_item)
            img2.thumbnail((180, 180), Image.ANTIALIAS)
            img2.save("resized_" + img_item, optimize = True, quality = 100)

        else:
            directory = r'C:\\Users\\kathe\Desktop\\FYP\\LeeYanJie\\FYP2\\UI-Images'
            os.chdir(directory)
            img_item = 'user_img.png'
            img2 = Image.open(img_item)
            img2.thumbnail((180, 180), Image.ANTIALIAS)
            img2.save("resized_" + img_item, optimize = True, quality = 100)
            
        img2 = Image.open("resized_" + img_item)
        img2 = ImageTk.PhotoImage(img2, Image.ANTIALIAS)
        profile_pic = Label(profile, image = img2, borderwidth = 0)
        profile_pic.place(x = 250, y = 80, anchor = "n")

        directory = r'C:\\Users\\kathe\\Desktop\\FYP\\LeeYanJie'
        os.chdir(directory)

        con = sqlite3.connect('attendance.db')

        cur = con.cursor()
        cur.execute("SELECT * FROM record")
        records = cur.fetchall()

        for record in records:
            if str(id) == str(record[1]):
                user_dc = str(record[3])
                user_atd = str(record[4])

        con.commit()
        con.close()
        user_DC_lbl = Label(profile, text = "Attendance Code", # Profile: User Attendance Code
                                bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
        user_DC_lbl.place(x = 50, y = space + 225)

        user_DC_txt = Label(profile, text = user_dc,
                                bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
        user_DC_txt.place(x = 200, y = space + 225)

        user_atd_lbl = Label(profile, text = "Attendance Status", # Profile: User Attendance Status
                                bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
        user_atd_lbl.place(x = 50, y = space + 250)

        user_atd_txt = Label(profile, text = user_atd,
                                bg = '#FAEEE0', font = ('helvetica', 10, 'bold'))
        user_atd_txt.place(x = 200, y = space + 250)

    # Create Change Password Button
    status = "change_psw_1"
    btn_change_pwd = Button(profile, text = "Change " + user_type + " Password",
                               command = lambda: updateRecord(profile, status, user_type, number, id),
                               bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3,
                               relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
    if number == 1:
        btn_change_pwd.place(x = 250, y = space + 280, anchor = "n")
    else:
        btn_change_pwd.place(x = 250, y = space + 230, anchor = "n")

    if number == 1:
        # Create Manual Attendance Button
        btn_attendance_sdt = Button(profile, text = "Manual Attendance",
                                    command = manualAttendance,
                                    bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3,
                                    relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
        btn_attendance_sdt.place(x = 250, y = space + 315, anchor = "n")

        # Create Image Capturing Button
        btn_attendance_sdt = Button(profile, text = "Train Images",
                                    command = lambda: addImage(user_name, user_id),
                                    bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3,
                                    relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
        btn_attendance_sdt.place(x = 250, y = space + 350, anchor = "n")
    
    elif number == 2:
        # Create View Attendance Button
        btn_attendance_stf = Button(profile, text = "View Attendance List",
                                    command = lambda: viewAttendance(profile),
                                    bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3,
                                    relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
        btn_attendance_stf.place(x = 250, y = space + 275, anchor = "n")

    # Exit Protocol
    profile.wm_protocol("WM_DELETE_WINDOW", lambda: exit(profile, panel))


# Take Image Panel - Student
def addImage(user_name, user_id):
    # Face Detection - Haar Cascade
    face_cascade = cv2.CascadeClassifier('FYP2/cascades/data/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Image Directory
    try:
        directory = r'C:\\Users\\kathe\Desktop\\FYP\\LeeYanJie\\FYP2\\Images\\' + user_id
        os.chdir(directory)

        # Initialize Individual Sampling Face Count
        lst = os.listdir(directory) # your directory path
        count = len(lst) + 1

        while(True):
            if count > 300:  # Take 50 Face Sample and Stop Video
                break

            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor = 1.05, minNeighbors = 6, minSize = (200, 200))

            for (x, y, w, h) in faces:
                roi_gray = gray[y: y + h, x: x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Save the Captured Image into the Datasets Folder
                img_item = (user_name.replace(" ", "_") + "_(" + str(count) + ").jpg" )

                cv2.imwrite(img_item, roi_gray)
                cv2.imshow('image', frame)
                count += 1
                print(count)

        # FYP Directory
        directory = r'C:\\Users\\kathe\\Desktop\\FYP\\LeeYanJie'
        os.chdir(directory)

        cap.release()
        cv2.destroyAllWindows()

    except:
        messagebox.showerror('Error', 'User Not Defined')


# Manual Attendance Panel - Student (Not Working)
def manualAttendance():
    return


# Take Attendance Panel - Student
def takeAttendance(user_name, user_id):
    check = False  # Check For Empty Input Field
    warn = ""  # Error Message Will Be Specificly Displayed

    con = sqlite3.connect('attendance.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM record")
    records = cur.fetchall()

    for record in records:
        if (user_name.get() == record[0] and user_id.get() == str(record[1])):
            attendance = str(record[4])
            id = str(record[1])
            if attendance == 'Present':
                warn = "You have already taken your attendance!"
            else:
                check = True
                attendance = 'Present'
                print(record)

    cur.execute("""
        UPDATE record SET 
        attendance = :attendance

        WHERE studentID = :id""",
        {
            'attendance': attendance, 'id': id
        })
    
    con.commit()
    con.close()

    if check == False:
        messagebox.showerror('Error', warn)
    else:
        messagebox.showinfo('Confirmation', 'Attendance for ' + id + ' is Marked ' + attendance + '!')


# View Attendance Panel - Staff
def viewAttendance(panel):
    panel.withdraw()
    view_atd = Toplevel()
    view_atd.title('Database Portal')
    view_atd.geometry('850x325')
    view_atd.configure(background = '#FAEEE0')

    try:  # Select Which Database to Output
        con = sqlite3.connect('attendance.db')

        # Create Update Digit Code Button
        Button(view_atd, text = "Refresh All Student's Digit Code", command = lambda: update_atd_all(view_atd, panel,1), bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3, relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold')).grid(row = 0, column = 0, padx = (20, 0), pady = 20)
        
        # Create Update Attendance Button
        Button(view_atd, text = "Mark All Student's Attendance", command = lambda: update_atd_all(view_atd, panel, 2), bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3, relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold')).grid(row = 0, column = 1, padx = (20, 0), pady = 20)

        # Using Treeview Widget
        trv = ttk.Treeview(view_atd, selectmode = 'browse')
        trv.grid(row = 1, rowspan = 10, column = 0, columnspan = 2, padx = (20, 5), sticky = 'nsew')

        # Number of Columns
        trv["columns"] = ("1", "2", "3", "4", "5", "6")
        
        # Defining Heading
        trv['show'] = 'headings'
        
        # Width of Columns and Alignment 
        trv.column("1", width = 50, anchor = 'c')
        trv.column("2", width = 150, anchor = 'c')
        trv.column("3", width = 100, anchor = 'c')
        trv.column("4", width = 250, anchor = 'c')
        trv.column("5", width = 100, anchor = 'c')
        trv.column("6", width = 100, anchor = 'c')
        
        # Headings for Respective Columns
        trv.heading("1", text = "No.")
        trv.heading("2", text = "Student Name")
        trv.heading("3", text = "Student ID")
        trv.heading("4", text = "Student Email")
        trv.heading("5", text = "Digit Code")
        trv.heading("6", text = "Attendance")

        cur = con.cursor()
        cur.execute("SELECT * FROM record")
        records = cur.fetchall()

        # Loop through Database Records
        noCount = 1
        for record in records:
            trv.insert("", 'end', iid = record[0], text = record[0], values = (noCount, record[0], record[1], record[2], record[3], record[4]))
            noCount += 1

        # Select User Event
        def item_selected(event):
            for selected_item in trv.selection():
                item = trv.item(selected_item)
                record = item['values']
                staffControl(view_atd, panel, record[2])

        # Select User
        trv.bind('<<TreeviewSelect>>', item_selected)

        # Add a Scrollbar
        scrollbar = ttk.Scrollbar(view_atd, orient = tkinter.VERTICAL, command = trv.yview)
        trv.configure(yscroll = scrollbar.set)
        scrollbar.grid(row = 1, rowspan = 10, column = 2, sticky = 'ns')

        con.commit()
        con.close()

    except Exception as ep:
        messagebox.showerror('', ep)

    # Exit Protocol
    view_atd.wm_protocol(
        "WM_DELETE_WINDOW", lambda: exit(view_atd, panel))


# Message Box Panel - Staff
def staffControl(view_atd, panel, id):
    view_atd.withdraw()
    win = Toplevel()
    win.title('Action Available')
    win.geometry('600x300')
    win.configure(background = '#FAEEE0')

    message = "  Would you like to make changes to the Database?  "
    Label(win, text = message, bg = '#FAEEE0', font = ('helvetica', 15, 'bold underline')).place(x = 50, y = 30)

    # Create Update Digit Code Button
    Button(win, text = "Update Digit Code", command = lambda: update_atd_one(win, panel, 1, id), bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3, relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold')).place(x = 100, y = 100)
    
    # Create Delete Attendance Button
    Button(win, text = "Refresh Attendance", command = lambda: update_atd_one(win, panel, 2, id), bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3, relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold')).place(x = 100, y = 150)
    
    # Create Cancel Button
    Button(win, text = "Cancel", command = lambda: exit(win, view_atd), bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3, relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold')).place(x = 100, y = 200)

    # Exit Protocol
    win.wm_protocol("WM_DELETE_WINDOW", lambda: exit(win, view_atd))


# Update Student Digit Code // Attendance Function
def update_atd_all(view_atd, panel, cmd_atd):
    con = sqlite3.connect('attendance.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM record")
    records = cur.fetchall()

    if cmd_atd == 1: 
        ans = messagebox.askyesno('Confirmation','Are you sure to refresh all digit code?')
        if ans == True:
            while True:
                for record in records:
                    pin = random.randint(100000, 999999)
                    if pin == record[3]:
                        continue
                    else:
                        cur.execute("""
                        UPDATE record SET 
                        digitCode = :digitCode

                        WHERE studentID = :id""",
                        {
                            'digitCode': pin, 'id': str(record[1])
                        })
                break

            con.commit()
            con.close()
            messagebox.showinfo('Confirmation', 'All Digit Code has Successfuly been Refreshed!')
        
        else:
            return

    elif cmd_atd == 2:
        ans = messagebox.askyesnocancel('Attendance', 'Click Yes to mark All Student Present and No to mark All Student Absent!')
        cancel = False
        if ans == True:
            attendance = "Present"
        elif ans == False:
            attendance = "Absent"
        else:
            cancel = True
        
        if cancel == False:
            for record in records:
                cur.execute("""
                UPDATE record SET 
                attendance = :attendance

                WHERE studentID = :id""",
                {
                    'attendance': attendance, 'id': str(record[1])
                })

            con.commit()
            con.close()
            messagebox.showinfo('Confirmation', 'All Student is now ' + attendance + '!')
    
    else:
        messagebox.showerror('Error', 'Action Not Defined')
    
    view_atd.withdraw()
    panel.deiconify()


# Update Student Digit Code // Attendance Function
def update_atd_one(win, panel, cmd_atd, id):
    con = sqlite3.connect('attendance.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM record")
    records = cur.fetchall()

    if cmd_atd == 1:
        while True:
            pin = random.randint(100000, 999999)
            for record in records:
                if pin == record[3]:
                    continue
                else:
                    cur.execute("""
                    UPDATE record SET 
                    digitCode = :digitCode

                    WHERE studentID = :id""",
                    {
                        'digitCode': pin, 'id': str(id)
                    })
            break

        con.commit()
        con.close()
        messagebox.showinfo('Confirmation', 'Digit Code for ' + str(id) + ' is Refreshed!')

    elif cmd_atd == 2:
        ans = messagebox.askyesno('Attendance', 'Would you like to change Attendance for ' + str(id) + '?')
        for record in records:
            if str(id) == str(record[1]):
                attendance = str(record[4])
                if ans == True:
                    if attendance == 'Present':
                        attendance = 'Absent'
                    else:
                        attendance = 'Present'

                cur.execute("""
                UPDATE record SET 
                attendance = :attendance

                WHERE studentID = :id""",
                {
                    'attendance': attendance, 'id': str(id)
                })

        con.commit()
        con.close()
        messagebox.showinfo('Confirmation', 'Attendance for ' + str(id) + ' is Marked ' + attendance + '!')          
    
    else:
        messagebox.showerror('Error', 'Action Not Defined')
    
    win.withdraw()
    panel.deiconify()

        
# Admin Panel
def admin(login, id):
    admin = Toplevel()
    admin.title('Admin Portal')
    admin.geometry('440x500')
    admin.configure(background = '#F9CF93')

    number = ""
    user_type = StringVar()
    user_type.set("User Category")
    options = ["Student", "Staff", "Admin"]

    # Update Toplevel Window Function
    def update(value):
        if value == "Student":
            user = "student"
            number = 1
            id = generateID(number)
        elif value == "Staff":
            user = "staff"
            number = 2
            id = generateID(number)
        else:
            user = "admin"
            number = 3
            id = generateID(number)

        user_id.config(state = 'normal')
        user_id.delete(0, END)
        user_id.insert(0, id)
        user_id.config(state = 'disabled')
        user_email.delete(0, END)
        user_email.insert(0, "@" + user + ".tarc.edu.my")

    # Create Textbox Labels
    title_lbl = Label(admin, text = "  Register New Member  ",
                      bg = '#F9CF93', font = ('helvetica', 15, 'bold underline'))
    title_lbl.place(x = 220, y = 30, anchor = "n")

    user_name_lbl = Label(admin, text = "User Name",
                          bg = '#F9CF93', font = ('helvetica', 10, 'bold'))
    user_name_lbl.place(x = 50, y = 125)

    user_id_lbl = Label(admin, text = "User ID",
                        bg = '#F9CF93', font = ('helvetica', 10, 'bold'))
    user_id_lbl.place(x = 50, y = 150)

    user_contact_lbl = Label(admin, text = "Contact Number",
                             bg = '#F9CF93', font = ('helvetica', 10, 'bold'))
    user_contact_lbl.place(x = 50, y = 175)

    user_gender_lbl = Label(admin, text = "Gender",
                            bg = '#F9CF93', font = ('helvetica', 10, 'bold'))
    user_gender_lbl.place(x = 50, y = 200)

    user_email_lbl = Label(admin, text = "Email Address",
                           bg = '#F9CF93', font = ('helvetica', 10, 'bold'))
    user_email_lbl.place(x = 50, y = 225)

    user_password_lbl = Label(admin, text = "Password",
                              bg = '#F9CF93', font = ('helvetica', 10, 'bold'))
    user_password_lbl.place(x = 50, y = 250)

    user_password_cfm_lbl = Label(admin, text = "Confirm Password",
                                  bg = '#F9CF93', font = ('helvetica', 10, 'bold'))
    user_password_cfm_lbl.place(x = 50, y = 275)

    # Create Input Widget
    user_type_ddm = OptionMenu(admin, user_type, *options, command = update)  # Drop Down Menu
    user_type_ddm.config(width = 20)
    user_type_ddm.config(bg = '#DBD0C0')
    user_type_ddm.config(activebackground = '#F9E4C8')
    user_type_ddm.config(font = ('helvetica', 10, 'bold'))
    user_type_ddm.place(x = 125, y = 80)

    user_name = Entry(admin, width = 30)
    user_name.place(x = 200, y = 125)

    user_id = Entry(admin, width = 30)
    user_id.place(x = 200, y = 150)

    user_contact = Entry(admin, width = 30)
    user_contact.place(x = 200, y = 175)

    user_gender = StringVar()  # Radio Button Variable
    user_gender.set("M")

    Radiobutton(admin, text = "Male", variable = user_gender, value = "M",
                bg = '#F9CF93', activebackground = '#F9CF93',
                font = ('helvetica', 10, 'bold')).place(x = 200, y = 200)

    Radiobutton(admin, text = "Female", variable = user_gender, value = "F",
                bg = '#F9CF93', activebackground = '#F9CF93',
                font = ('helvetica', 10, 'bold')).place(x = 270, y = 200)

    user_email = Entry(admin, width = 30)
    user_email.place(x = 200, y = 225)

    user_password = Entry(admin, width = 30, show = '*')
    user_password.place(x = 200, y = 250)

    user_password_cfm = Entry(admin, width = 30, show = '*')
    user_password_cfm.place(x = 200, y = 275)

    # Create Register Button
    status = "new_user"
    old_id = ""
    btn_register_user = Button(admin, text = "Add Record to Database",
                               command = lambda: insertRecord
                               (admin, admin, status, user_type, number, user_name, user_id, user_contact,
                                user_gender, user_email, user_password, user_password_cfm, old_id),
                               bg = '#DBD0C0', width = 32, height = 1, borderwidth = 3,
                               relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
    btn_register_user.place(x = 220, y = 320, anchor = "n")

    # Create View Student Database Button
    btn_query_sdt = Button(admin, text = "View Student Records",
                           command = lambda: queryRecord(admin, 1),
                           bg = '#DBD0C0', width = 32, height = 1, borderwidth = 3,
                           relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
    btn_query_sdt.place(x = 220, y = 360, anchor = "n")

    # Create View Staff Database Button
    btn_query_stf = Button(admin, text = "View Staff Records",
                           command = lambda: queryRecord(admin, 2),
                           bg = '#DBD0C0', width = 32, height = 1, borderwidth = 3,
                           relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
    btn_query_stf.place(x = 220, y = 400, anchor = "n")

    # Create View Staff Database Button
    btn_query_adn = Button(admin, text = "View Admin Records",
                           command = lambda: queryRecord(admin, 3),
                           bg = '#DBD0C0', width = 32, height = 1, borderwidth = 3,
                           relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
    btn_query_adn.place(x = 220, y = 440, anchor = "n")


    # Exit Protocol
    admin.wm_protocol("WM_DELETE_WINDOW", lambda: exit(admin, login))


# View User Database Panel - Admin
def queryRecord(admin, number):
    admin.withdraw()
    admin_record = Toplevel()
    admin_record.title('Database Portal')
    admin_record.geometry('1000x500')
    admin_record.configure(background = '#FAEEE0')
    admin_record.grid_rowconfigure(0, weight = 1)
    admin_record.grid_columnconfigure(0, weight = 1)

    try:  # Select Which Database to Output
        if number == 1:
            user_type = "Student"
            con = sqlite3.connect('studentdata.db')
        elif number == 2:
            user_type = "Staff"
            con = sqlite3.connect('staffdata.db')
        elif number == 3:
            user_type = "Admin"
            con = sqlite3.connect('admindata.db')
        else:
            messagebox.showerror('Error', 'User Not Defined')
     
        # Using Treeview Widget
        trv = ttk.Treeview(admin_record, selectmode = 'browse')
        trv.grid(row = 0, column = 0, pady = 20, sticky = 'nsew')

        # Number of Columns
        trv["columns"] = ("1", "2", "3","4","5", "6", "7")
        
        # Defining Heading
        trv['show'] = 'headings'
        
        # Width of Columns and Alignment 
        trv.column("1", width = 50, anchor = 'c')
        trv.column("2", width = 150, anchor = 'c')
        trv.column("3", width = 100, anchor = 'c')
        trv.column("4", width = 150, anchor = 'c')
        trv.column("5", width = 100, anchor = 'c')
        trv.column("6", width = 250, anchor = 'c')
        trv.column("7", width = 150, anchor = 'c')
        
        # Headings for Respective Columns
        trv.heading("1", text = "No.")
        trv.heading("2", text = user_type + " Name")
        trv.heading("3", text = user_type + " ID")
        trv.heading("4", text = "Contact Number")
        trv.heading("5", text = "Gender")
        trv.heading("6", text = user_type + " Email")  
        trv.heading("7", text = "Password")

        cur = con.cursor()
        cur.execute("SELECT * FROM record")
        records = cur.fetchall()

        # Loop through Database Records
        noCount = 1
        for record in records:
            trv.insert("", 'end', iid = record[0], text = record[0],
                       values = (noCount, record[0], record[1], record[2], record[3], record[4], record[5]))
            noCount += 1

        # Select User Event
        def item_selected(event):
            for selected_item in trv.selection():
                item = trv.item(selected_item)
                record = item['values']
                adminControl(admin_record, admin, user_type, number, record[2])

        # Select User
        trv.bind('<<TreeviewSelect>>', item_selected)

        # Add a Scrollbar
        scrollbar = ttk.Scrollbar(admin_record, orient = tkinter.VERTICAL, command = trv.yview)
        trv.configure(yscroll = scrollbar.set)
        scrollbar.grid(row = 0, column = 1, pady = 20, sticky = 'ns')

        con.commit()
        con.close()

    except Exception as ep:
        messagebox.showerror('', ep)

    # Exit Protocol
    admin_record.wm_protocol(
        "WM_DELETE_WINDOW", lambda: exit(admin_record, admin))


# Message Box Panel - Admin
def adminControl(admin_record, admin, user_type, number, id):
    admin_record.withdraw()
    win = Toplevel()
    win.title('Action Available')
    win.geometry('600x300')
    win.configure(background = '#FAEEE0')

    message = "  Would you like to make changes to the Database?  "
    Label(win, text = message, bg = '#FAEEE0', font = ('helvetica', 15, 'bold underline')).place(x = 50, y = 30)

    # Create Update Record Button
    status = "admin_edit"
    Button(win, text = "Update " + user_type + " Info", command = lambda: switchWindow(win, admin, status, user_type, number, id), bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3, relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold')).place(x = 100, y = 100)
    
    # Create Delete Record Button
    Button(win, text = "Delete " + user_type, command = lambda: deleteRecord(win, admin, number, id), bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3, relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold')).place(x = 100, y = 150)
    
    # Create Cancel Button
    Button(win, text = "Cancel", command = lambda: exit(win, admin_record), bg = '#DBD0C0', width = 38, height = 1, borderwidth = 3, relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold')).place(x = 100, y = 200)

    # Exit Protocol
    win.wm_protocol("WM_DELETE_WINDOW", lambda: exit(win, admin_record))


# Delete User Info Function
def deleteRecord(win, admin, number, id):
    try:
        ans = messagebox.askyesno('Confirmation','Are you sure to delete record for User ID : ' + str(id) + ' ? ')
        if ans == True:
            if number == 1:            
                con = sqlite3.connect('studentdata.db')
                cur = con.cursor()
                
                try:
                    cur.execute("SELECT * FROM record")
                    records = cur.fetchall()
                    for record in records:
                        if id == record[1]:
                            file = str(record[1])

                    directory = r'C:\\Users\\kathe\Desktop\\FYP\\LeeYanJie\\FYP2\\Images'
                    os.chdir(directory)

                    for i in os.listdir(file):
                        os.remove(os.path.join(file, i))
                    os.rmdir(file)

                except:
                    messagebox.showerror('Opps', 'Image file not found')

                directory = r'C:\\Users\\kathe\\Desktop\\FYP\\LeeYanJie'
                os.chdir(directory)

                cur.execute("DELETE FROM record WHERE studentID = " + str(id))
                con.commit()

                con = sqlite3.connect('attendance.db')
                cur = con.cursor()
                cur.execute("DELETE FROM record WHERE studentID = " + str(id))
                
            elif number == 2:
                con = sqlite3.connect('staffdata.db')
                cur = con.cursor()
                cur.execute("DELETE FROM record WHERE staffID = " + str(id))

            elif number == 3:
                con = sqlite3.connect('admindata.db')
                cur = con.cursor()
                cur.execute("DELETE FROM record WHERE adminID = " + str(id))
            else:
                messagebox.showerror('Error', 'User Not Defined')
            
            con.commit()
            con.close()
            messagebox.showinfo('Confirmation', 'Record Deleted!')
            win.destroy()
            queryRecord(admin, number)

        else:
            return

    except Exception as ep:
        messagebox.showerror('', ep)


# Update User Info Panel
def updateRecord(panel, status, user_type, number, id):
    panel.withdraw()
    update_data = Toplevel()
    update_data.title('Record Update Portal')
    update_data.configure(background = '#DBD0C0')

    if status == "forget_psw":  # View in Login Panel
        update_data.geometry('600x400')

        # Create User Info Fill-in Panel
        title_lbl = Label(update_data, text = "  Password Recovering  ",
                        bg = '#DBD0C0', font = ('helvetica', 15, 'bold underline'))
        title_lbl.place(x = 300, y = 30, anchor = "n")
        
        descript_lbl = Label(update_data, text = "Fill in Personal Details to Get 6 Digit Code from Your Email Address.",
                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
        descript_lbl.place(x = 300, y = 75, anchor = "n")

        user_name_lbl = Label(update_data, text = "User Name",
                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
        user_name_lbl.place(x = 70, y = 110)
        
        user_name = Entry(update_data, width = 50)
        user_name.place(x = 220, y = 110)

        user_id_lbl = Label(update_data, text = "User ID",
                            bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
        user_id_lbl.place(x = 70, y = 135)

        user_id = Entry(update_data, width = 50)
        user_id.place(x = 220, y = 135)

        user_email_lbl = Label(update_data, text = "Email Address",
                            bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
        user_email_lbl.place(x = 70, y = 160)

        user_email = Entry(update_data, width = 50)
        user_email.place(x = 220, y = 160)
        user_email.delete(0, END)
        user_email.insert(0, "@" + user_type + ".tarc.edu.my")

        # Create Send Email Button        
        digit_code = random.randint(100000, 999999)
        btn_send_DC = Button(update_data, text = "Send Digit Code to Email Address",
                            command = lambda: verifyIdentity(number, user_name, user_id, user_email, digit_code),
                            bg = '#FAEEE0', width = 32, height = 1, borderwidth = 3,
                            relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
        btn_send_DC.place(x = 300, y = 200, anchor = "n")

        # Create Verification Panel
        line_break = line(76)

        line_break_lbl = Label(update_data, text = line_break,
                            bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
        line_break_lbl.place(x = 25, y = 240)

        descript_lbl = Label(update_data, text = "Enter 6 Digit Code to Proceed Re-setting Your New Password.",
                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
        descript_lbl.place(x = 300, y = 275, anchor = "n")

        psw_change_code_lbl = Label(update_data, text = "Digit Code (Email)",
                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
        psw_change_code_lbl.place(x = 70, y = 310)

        psw_change_code = Entry(update_data, width = 50)
        psw_change_code.place(x = 220, y = 310)

        # Create Reset Password Button
        btn_reset_password = Button(update_data, text = "Confirm Reset Password",
                               command = lambda: forgetPassword(update_data, user_type, number, user_id, digit_code, psw_change_code),
                               bg = '#FAEEE0', width = 32, height = 1, borderwidth = 3,
                               relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
        btn_reset_password.place(x = 300, y = 350, anchor = "n")

    else:
        try:  # Select Which Database to Output
            if number == 1:
                con = sqlite3.connect('studentdata.db')
                cur = con.cursor()
                cur.execute("SELECT * FROM record WHERE studentID = " + str(id))
            elif number == 2:
                con = sqlite3.connect('staffdata.db')
                cur = con.cursor()
                cur.execute("SELECT * FROM record WHERE staffID = " + str(id))
            elif number == 3:
                con = sqlite3.connect('admindata.db')
                cur = con.cursor()
                cur.execute("SELECT * FROM record WHERE adminID = " + str(id))

            records = cur.fetchall()

            if (status == "change_psw_1" or status == "change_psw_2"): # View in Profile Panel
                update_data.geometry('450x400')

                for record in records:
                    if str(id) == str(record[1]):
                        user_name = str(record[0])
                        user_id = str(record[1])
                        user_email = str(record[4])
                        current_psw = str(record[5])

                # Create User Info Labels
                title_lbl = Label(update_data, text = "  Update Your Password  ",
                                bg = '#DBD0C0', font = ('helvetica', 15, 'bold underline'))
                title_lbl.place(x = 225, y = 30, anchor = "n")

                user_name_lbl = Label(update_data, text = user_type + " Name",
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_name_lbl.place(x = 50, y = 100)

                user_id_lbl = Label(update_data, text = user_type + " ID",
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_id_lbl.place(x = 50, y = 125)

                user_email_lbl = Label(update_data, text = "Email Address",
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_email_lbl.place(x = 50, y = 150)

                user_name_txt = Label(update_data, text = user_name,
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_name_txt.place(x = 200, y = 100)
                
                user_ID_txt = Label(update_data, text = user_id,
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_ID_txt.place(x = 200, y = 125)

                user_email_txt = Label(update_data, text = user_email,
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_email_txt.place(x = 200, y = 150)                

                line_break = line(55)

                line_break_lbl = Label(update_data, text = line_break,
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                line_break_lbl.place(x = 25, y = 175)   

                # Create New Password Input             
                user_password_lbl = Label(update_data, text = "Current Password",
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_password_lbl.place(x = 50, y = 225)

                user_password_new_lbl = Label(update_data, text = "New Password",
                                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_password_new_lbl.place(x = 50, y = 250)

                user_password_cfm_lbl = Label(update_data, text = "Confirm Password",
                                            bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_password_cfm_lbl.place(x = 50, y = 275)

                user_password = Entry(update_data, width = 30, show = '*')
                user_password.place(x = 200, y = 225)

                if status == "change_psw_2":
                    user_password.config(state = 'normal')
                    user_password.delete(0, END)
                    user_password.insert(0, current_psw)
                    user_password.config(state = 'disabled')

                user_password_new = Entry(update_data, width = 30, show = '*')
                user_password_new.place(x = 200, y = 250)

                user_password_cfm = Entry(update_data, width = 30, show = '*')
                user_password_cfm.place(x = 200, y = 275)
                
                # Create Change Password Button
                btn_change_password = Button(update_data, text = "Confirm Password Change",
                                        command = lambda: insertPassword
                                        (update_data, status, number, id, user_password, user_password_new, user_password_cfm),
                                        bg = '#FAEEE0', width = 32, height = 1, borderwidth = 3,
                                        relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
                btn_change_password.place(x = 225, y = 325, anchor = "n")

            elif status == "admin_edit": # View in Database Panel
                update_data.geometry('450x400')

                # Create Textbox Labels
                title_lbl = Label(update_data, text = "  Update " + user_type + " Information  ",
                                bg = '#DBD0C0', font = ('helvetica', 15, 'bold underline'))
                title_lbl.place(x = 225, y = 30, anchor = "n")

                user_name_lbl = Label(update_data, text = user_type + " Name",
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_name_lbl.place(x = 50, y = 125)

                user_id_lbl = Label(update_data, text = user_type + " ID",
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_id_lbl.place(x = 50, y = 150)

                user_contact_lbl = Label(update_data, text = "Contact Number",
                                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_contact_lbl.place(x = 50, y = 175)

                user_gender_lbl = Label(update_data, text = "Gender",
                                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_gender_lbl.place(x = 50, y = 200)

                user_email_lbl = Label(update_data, text = "Email Address",
                                    bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_email_lbl.place(x = 50, y = 225)

                user_password_lbl = Label(update_data, text = "Password",
                                        bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_password_lbl.place(x = 50, y = 250)

                user_password_cfm_lbl = Label(update_data, text = "Confirm Password",
                                            bg = '#DBD0C0', font = ('helvetica', 10, 'bold'))
                user_password_cfm_lbl.place(x = 50, y = 275)

                # Create Input Widget
                user_name = Entry(update_data, width = 30)
                user_name.place(x = 200, y = 125)

                user_id = Entry(update_data, width = 30)
                user_id.place(x = 200, y = 150)

                user_contact = Entry(update_data, width = 30)
                user_contact.place(x = 200, y = 175)

                global user_gender
                user_gender = StringVar()  # Radio Button Variable
                for record in records:
                    if record[3] == "M":
                        user_gender.set("M")
                    else:
                        user_gender.set("F")

                Radiobutton(update_data, text = "Male", variable = user_gender, value = "M",
                            bg = '#DBD0C0', activebackground = '#DBD0C0',
                            font = ('helvetica', 10, 'bold')).place(x = 200, y = 200)

                Radiobutton(update_data, text = "Female", variable = user_gender, value = "F",
                            bg = '#DBD0C0', activebackground = '#DBD0C0',
                            font = ('helvetica', 10, 'bold')).place(x = 270, y = 200)

                user_email = Entry(update_data, width = 30)
                user_email.place(x = 200, y = 225)

                user_password = Entry(update_data, width = 30, show = '*')
                user_password.place(x = 200, y = 250)

                user_password_cfm = Entry(update_data, width = 30, show = '*')
                user_password_cfm.place(x = 200, y = 275)

                #Loop through Database Records
                for record in records:
                    user_name.insert(0, record[0])
                    user_id.insert(0, record[1])
                    user_contact.insert(0, record[2])
                    user_email.insert(0, record[4])
                    user_password.insert(0, record[5])

                    id = record[1]
                
                user_id.config(state = 'disabled')

                # Create Update Button
                btn_update_user = Button(update_data, text = "Submit Record",
                                        command = lambda: insertRecord
                                        (update_data, panel, status, user_type, number, user_name, user_id, user_contact, user_gender, user_email, user_password, user_password_cfm, id),
                                        bg = '#FAEEE0', width = 32, height = 1, borderwidth = 3,
                                        relief = 'ridge', activebackground = '#F9E4C8', font = ('helvetica', 12, 'bold'))
                btn_update_user.place(x = 225, y = 325, anchor = "n")   

        except Exception as ep:
            messagebox.showerror('', ep)
        
    # Exit Protocol
    update_data.wm_protocol("WM_DELETE_WINDOW", lambda: exit(update_data, panel))


# Register // Edit User Info Function
def insertRecord(panel_old, panel_new, status, user_type, number, user_name, user_id, user_contact, user_gender, user_email, user_password, user_password_cfm, id):
    check = False # Check For Empty Input Field
    warn = "" # Error Message Will Be Specificly Displayed

    if status == "new_user":
        user_type = str(user_type.get())
        if user_type == "Student":
            number = 1
        if user_type == "Staff":
            number = 2
        if user_type == "Admin":
            number = 3

    if user_name.get() == "":
        warn = "User Name can't be empty!"
    elif user_contact.get() == "":
        warn = "Contact Number can't be empty!"
    elif user_gender.get() == "":
        warn = "Select Gender"
    elif user_email.get() == "":
        warn = "User Email can't be empty!"
    elif user_password.get() == "":
        warn = "Password can't be empty!"
    elif user_password_cfm.get() == "":
        warn = "Re-entered Password can't be empty!"
    elif user_password.get() != user_password_cfm.get():
        warn = "Passwords didn't match!"
    else:
        check = True
        try:
            if number == 1: # Drop Down Option --> Student
                con = sqlite3.connect('studentdata.db')
                cur = con.cursor()

                if status == "admin_edit":
                    cur.execute(
                        "DELETE FROM record WHERE studentID = " + str(id))

                cur.execute(
                    "INSERT INTO record VALUES (:name, :studentID, :contact, :gender, :email, :password)",
                    {
                        'name': user_name.get(),
                        'studentID': user_id.get(),
                        'contact': user_contact.get(),
                        'gender': user_gender.get(),
                        'email': user_email.get(),
                        'password': user_password.get()
                    }
                )
                con.commit()

                attendance = "Absent"
                pin = ""

                if status == "new_user":
                    generateCode(user_name, user_id, user_email, pin, attendance)
                    directory = r'C:\\Users\\kathe\Desktop\\FYP\\LeeYanJie\\FYP2\\Images'
                    os.chdir(directory)

                    os.makedirs(str(user_id.get()), exist_ok = True)

                    directory = r'C:\\Users\\kathe\\Desktop\\FYP\\LeeYanJie'
                    os.chdir(directory)
                    
                elif status == "admin_edit":
                    con = sqlite3.connect('attendance.db')
                    cur = con.cursor()
                    cur.execute(
                        "DELETE FROM record WHERE studentID = " + str(id))
                                        
                    cur.execute("SELECT * FROM record")
                    records = cur.fetchall()                
                        
                    while True:
                        pin = random.randint(100000, 999999)
                        for record in records:
                            if pin == record[3]:
                                continue

                        cur.execute(
                            "INSERT INTO record VALUES (:studentName, :studentID, :studentEmail, :digitCode, :attendance)",
                            {
                                'studentName': user_name.get(),
                                'studentID': user_id.get(),
                                'studentEmail': user_email.get(),
                                'digitCode': pin,
                                'attendance': attendance
                            })

                        ### Update File Name
                        directory = r'C:\\Users\\kathe\Desktop\\FYP\\LeeYanJie\\FYP2\\Images\\' + user_id.get() + '\\' 
                        os.chdir(directory)
                        count = 1
                        keyword = "resized_"
                        
                        for file_name in os.listdir(directory):
                            keyword2 = '(' + str(count) + ')'

                            for file_name in os.listdir(directory):
                                # Construct old File Name
                                if keyword2 in file_name:
                                    source = directory + file_name
                                    #Construct New File Name
                                    if keyword in file_name:
                                        destination = directory + keyword + (user_name.get()).replace(" ", "_") + "_(" + str(count) + ").jpg"
                                    else:
                                        destination = directory + (user_name.get()).replace(" ", "_") + "_(" + str(count) + ").jpg"

                            # Renaming the file
                            try:
                                os.rename(source, destination)
                                count += 1
                                print(count)
                            except:
                                break
                            
                        directory = r'C:\\Users\\kathe\\Desktop\\FYP\\LeeYanJie'
                        os.chdir(directory)
                        break
                    
            elif number == 2: # Drop Down Option --> Staff
                con = sqlite3.connect('staffdata.db')
                cur = con.cursor()
                if status == "admin_edit":
                    cur.execute(
                        "DELETE FROM record WHERE staffID = " + str(id))

                cur.execute(
                    "INSERT INTO record VALUES (:name, :staffID, :contact, :gender, :email, :password)",
                    {
                        'name': user_name.get(),
                        'staffID': user_id.get(),
                        'contact': user_contact.get(),
                        'gender': user_gender.get(),
                        'email': user_email.get(),
                        'password': user_password.get()
                    }
                )

            elif number == 3: # Drop Down Option --> Admin
                con = sqlite3.connect('admindata.db')
                cur = con.cursor()
                if status == "admin_edit":
                    cur.execute(
                        "DELETE FROM record WHERE adminID = " + str(id))

                cur.execute(
                    "INSERT INTO record VALUES (:name, :adminID, :contact, :gender, :email, :password)",
                    {
                        'name': user_name.get(),
                        'adminID': user_id.get(),
                        'contact': user_contact.get(),
                        'gender': user_gender.get(),
                        'email': user_email.get(),
                        'password': user_password.get()
                    }
                )

            con.commit()
            con.close()
            messagebox.showinfo('Confirmation', 'Record Saved!')
            panel_old.withdraw()
            panel_new.deiconify()

        except Exception as ep:
            messagebox.showerror('', ep)

    if check == False:
        messagebox.showerror('Error', warn)


# Change // Forget Password Function
def insertPassword(panel, status, number, id, user_password, user_password_new, user_password_cfm):
    check = False # Check For Empty Input Field
    warn = "" # Error Message Will Be Specificly Displayed

    try:
        if number == 1: # Drop Down Option --> Student
            con = sqlite3.connect('studentdata.db')
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM record WHERE studentID = " + str(id))
        if number == 2: # Drop Down Option --> Staff
            con = sqlite3.connect('staffdata.db')
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM record WHERE staffID = " + str(id))
        if number == 3: # Drop Down Option --> Admin
            con = sqlite3.connect('admindata.db')
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM record WHERE adminID = " + str(id))

        records = cur.fetchall()

        # Pass Current Password to User
        for record in records:
            if status == "forget_psw":
                user_password = record[5]            
            
            # Check Validity of Current Password
            if user_password.get() == "":
                warn = "Current Password can't be empty!"
            elif record[5] != user_password.get():
                warn = "Current Password incorrect!"                
            elif user_password_new.get() == "":
                warn = "New Password can't be empty!"
            elif user_password_cfm.get() == "":
                warn = "Re-entered Password can't be empty!"
            elif user_password.get() == user_password_new.get():
                warn = "New Password can't be same as Current Password!"
            elif user_password_new.get() != user_password_cfm.get():
                warn = "Passwords didn't match!"
            else:
                check = True
                if number == 1:
                    cur.execute("""
                    UPDATE record SET
                    password = :password

                    WHERE studentID = :id""",
                    {
                        'password': user_password_new.get(), 'id': str(id)
                    })

                elif number == 2:
                    cur.execute("""
                    UPDATE record SET
                    password = :password

                    WHERE staffID = :id""",
                    {
                        'password': user_password_new.get(), 'id': str(id)
                    })

                elif number == 3:
                    cur.execute("""
                    UPDATE record SET
                    password = :password

                    WHERE adminID = :id""",
                    {
                        'password': user_password_new.get(), 'id': str(id)
                    })

                con.commit()
                con.close()
                messagebox.showinfo('Confirmation', 'Record Saved!')
                panel.withdraw()
                root.deiconify()

    except Exception as ep:
        messagebox.showerror('', ep)
        
    if check == False:
        messagebox.showerror('Error', warn)
        

# Separating Line Function
def line(number):
    new_string = "_"

    for char in new_string:
        new_string = new_string + char * number

    return new_string


# Switch Window - Admin (for Update Record)
def switchWindow(win, admin, status, user_type, number, id):
    win.destroy()
    updateRecord(admin, status, user_type, number, id)


# Exit Toplevel Window Function
def exit(panel_old, panel_new):
    panel_old.withdraw()
    panel_new.deiconify()


# Exit Program Function
def on_closing():
    if messagebox.askyesno("Quit", "Do you want to quit?"):
        root.destroy()


# database()
# facesTrain()

# Create Facial Recognition Button
btn2 = Image.open(
    "C:/Users/kathe/Desktop/FYP/LeeYanJie/FYP2/UI-images/camera_img.jpg")
btn2 = btn2.resize((100, 100), Image.ANTIALIAS)
btn2 = ImageTk.PhotoImage(btn2)

btn_face_recog = Button(root, image = btn2,
                        command = confirmIdentity,
                        # command = faces('Lee Yan Jie', 2005661),
                        borderwidth = 0)
btn_face_recog.place(x = 50, y = 20)

# Display Logo Image
img1 = Image.open(
    "C:/Users/kathe/Desktop/FYP/LeeYanJie/FYP2/UI-images/tarc_img.jpg")
img1 = img1.resize((450, 150), Image.ANTIALIAS)
img1 = ImageTk.PhotoImage(img1)

tarc_logo = Label(root, image = img1, borderwidth = 0)
tarc_logo.place(x = 500, y = 20, anchor = "n")

# Create Admin Panel Button
btn1 = Image.open(
    "C:/Users/kathe/Desktop/FYP/LeeYanJie/FYP2/UI-images/admin_img.jpg")
btn1 = btn1.resize((100, 100), Image.ANTIALIAS)
btn1 = ImageTk.PhotoImage(btn1)

btn_login_adn = Button(root, image = btn1,
                       command = lambda: homeLogin(3),
                       borderwidth = 0)
btn_login_adn.place(x = 850, y = 20)

# Create Student Panel Button
btn_login_sdt = Button(root, text = "Login as Student",
                       command = lambda: homeLogin(1),
                       bg = '#F9CF93', width = 30, height = 2, borderwidth = 3,
                       relief = 'ridge', activebackground = '#F9E4C8', font = ('times', 20, 'bold'))
btn_login_sdt.place(x = 500, y = 190, anchor = "n")

# Create Staff Panel Button
btn_login_stf = Button(root, text = "Login as Staff",
                       command = lambda: homeLogin(2),
                       bg = '#F9CF93', width = 30, height = 2, borderwidth = 3,
                       relief = 'ridge', activebackground = '#F9E4C8', font = ('times', 20, 'bold'))
btn_login_stf.place(x = 500, y = 300, anchor = "n")

# Exit Protocol
# root.protocol("WM_DELETE_WINDOW", on_closing)


root.mainloop()
