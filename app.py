from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
# from flask_socketio import SocketIO
from speech_recognition import app as speech_app  # Import the speech recognition app
from chatbot import *
import pandas as pd
import mysql.connector
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

app = Flask(__name__)
socketio = SocketIO(app)


# Define global variables
global clf, cols, current_symptom_index

# Load data
getSeverityDict()
getDescription()
getprecautionDict()

# Load the classifier and feature columns
training = pd.read_csv('Training.csv')
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
clf = DecisionTreeClassifier()
clf.fit(x, y)

current_symptom_index = 0  # In


server_name = "127.0.0.1"
username = "root"
password = ""
dbname = "digital_doctor"

# Create a connection
try:
    conn = mysql.connector.connect(
        host=server_name,
        user=username,
        passwd=password,
        database=dbname
    )
    print("Connected to the database successfully!")
except mysql.connector.Error as err:
    print(f"Error: {err}")
def fetch_user_data(user_id):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    return user
def speech_to_text():
    # You can add code here to process the speech-to-text request
    # For example, you can retrieve audio data from the request and process it using the speech recognition library
    # Once you have the text, you can return it as JSON or perform any additional processing
    return jsonify({'message': 'Speech-to-text functionality will be implemented here'})
# Secret key for session management (change this to a secure key in production)
app.secret_key = 'your_secret_key'

@app.route('/')
def login1():
    return render_template('login.html')
@app.route('/register')
def register():
    return render_template('register.html')
@app.route('/register_user', methods=['POST'])
def register_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        blood_group = request.form['bloodGroup']
        height = request.form['height']
        weight = request.form['weight']
        guardian_name = request.form['guardianName']

        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO users (username, password, blood_group, height, weight, guardian_name)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (username, password, blood_group, height, weight, guardian_name))
            conn.commit()
            flash('User registered successfully!', 'success')
            return redirect(url_for('login1'))
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            flash('Failed to register user. Please try again.', 'error')

    return render_template('register.html')
@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        users = cursor.fetchall()  # Fetch the result set
        cursor.close()  # Close the cursor after fetching the result set

        if users:
            user = users[0]  # Take the first user from the list
            session['user_id'] = user['id']
            # Redirect to index route after successful login
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials', 'error')
            return redirect(url_for('login1'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' in session:
        user_id = session['user_id']
        user = fetch_user_data(user_id)
        if user:
            return render_template('dashboard.html', user=user, fetch_user_data=fetch_user_data)
        else:
            flash('User not found', 'error')
            return redirect(url_for('login1'))
    else:
        flash('You need to log in first', 'error')
        return redirect(url_for('login1'))

@app.route('/loc')
def loc():
    return render_template('loc.html')
@app.route('/trial')
def trial():
    return render_template('index3.html')
@app.route('/vaccine')
def vaccine():
    return render_template('vaccine.html')
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login1'))
@app.route('/user_details')
def user_details():
    if 'user_id' in session:
        user_id = session['user_id']
        user = fetch_user_data(user_id)
        if user:
            user_details = {
                'blood_group': user['blood_group'],
                'height': user['height'],
                'weight': user['weight'],
                'guardian_name': user['guardian_name']
            }
            return jsonify(user_details)
    return jsonify({})
@app.route('/index')
def index():
    if 'user_id' in session:
        user_id = session['user_id']
        user = fetch_user_data(user_id)
        if user:
            return render_template('index2.html', user=user, fetch_user_data=fetch_user_data)
        else:
            flash('User not found', 'error')
            return redirect(url_for('login1'))
    else:
        flash('You need to log in first', 'error')
        return redirect(url_for('login1'))


user_symptom = None
num_days = None  # Initialize num_days to None initially
symptom_count = 0  # Track the number of symptoms provided by the user
@app.route('/send_message', methods=['POST'])
def send_message():
    global user_symptom, current_symptom_index, num_days, symptom_count  # Access the global variables
    
    if request.method == 'POST':
        message = request.form['message']
        user_name = request.form.get('name')  # Get user name from form
        
        if message.lower() in ["hi", "hello", "hey"]:
            chatbot_response = getInfo(user_name)  # Pass user name to getInfo function
        elif message.lower() in ["thanks", "okay","thank you"]:
            chatbot_response="Thank you have a nice day"
            exit
        elif user_symptom is None:
            # Extract and store the symptom entered by the user
            user_symptom = message.strip()
            symptom_count += 1  # Increment the symptom count
            if symptom_count == 1:  # Ask for the number of days only for the first symptom
                chatbot_response = "Please enter the number of days for which you've had this symptom."
            else:
                chatbot_response = tree_to_code(clf, cols, num_days, user_symptom, 0)
                # Reset the global variables after processing the first symptom
                user_symptom = None
                
        elif num_days is None:
            try:
                num_days = int(message)  # Try to convert the message to an integer
                chatbot_response = tree_to_code(clf, cols, num_days, user_symptom, 0)
                s = ""
                if len(chatbot_response) == 2:
                    s += chatbot_response[0]+" "+chatbot_response[1]
                    chatbot_response = s

                print("Function call tree done")

                #user_symptom = next_symptom 
                user_symptom = None  # Reset the global variable after processing
            except ValueError:
                chatbot_response = "Please enter a valid number of days."
        elif user_symptom:
            # Assuming the user has already entered a symptom
            try:
                if message.lower() == 'yes':
                    chatbot_response, next_symptom = tree_to_code(clf, cols, num_days, user_symptom, current_symptom_index)
                    user_symptom = next_symptom  # Store the next symptom for the next iteration
                    current_symptom_index += 1  # Increment the current symptom index
                else:
                    # User responded 'no', move to the next symptom
                    chatbot_response, next_symptom = tree_to_code(clf, cols, num_days, user_symptom, current_symptom_index)
                    user_symptom = next_symptom  # Store the next symptom for the next iteration
                    current_symptom_index += 1  # Increment the current symptom index
            except ValueError:
                chatbot_response = "Please enter a valid response (yes/no)."
        else:
            chatbot_response = "Please enter your symptom first."
        
        # Return the chatbot response
        return jsonify(chatbot_response)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
    speech_app.run(debug=True, port=5001)  # Assuming you want to run the speech recognition app on a different port
