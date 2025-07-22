import streamlit as st
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO
from textblob import TextBlob
import matplotlib.pyplot as plt
from groq import Groq

# Load credentials
load_dotenv()
API_KEY = os.getenv("VAPI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
PHONE_ID = os.getenv("PHONE_NUMBER_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
LOG_FILE = "call_logs.json"

# Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# QA Questions
qa_questions = [
    "Did the agent introduce themselves and identify the dealership during the introduction?",
    "Did the agent identify the reason for the call?",
    "Did the agent speak to the correct person?",
    "Did the agent verify the phone/email address?",
    "Did the agent verify the vehicle information?",
    "Did the agent ask for current mileage and was clear on campaign service (DFS/INA/Recall)?",
    "Did the agent recap the appointment and represent the dealership positively and with a sense of trust?",
    "Did the agent set urgency in scheduling an appointment as soon as possible?",
    "Was the agent professional throughout the call?",
    "Did the agent speak confidently?",
    "Did the agent display active listening skills?",
    "Did the agent sound upbeat, enthusiastic, and friendly?"
]

# Utility Functions
def save_log(entry):
    try:
        logs = json.load(open(LOG_FILE))
    except FileNotFoundError:
        logs = []
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def load_logs():
    try:
        return json.load(open(LOG_FILE))
    except:
        return []

def fetch_call_details(call_id):
    url = f"https://api.vapi.ai/call/{call_id}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.ok:
        try:
            return resp.json()
        except Exception:
            try:
                return json.loads(resp.text)
            except:
                st.error("Could not parse API response as JSON.")
                return None
    st.error("Failed to fetch call details.")
    return None

# Auth UI
def login():
    st.title("Login")
    email = st.text_input("Enter your Gmail address", key="email_input")
    if st.button("Login", key="login_button"):
        if email and email.endswith(".com"):
            st.session_state.user = email
            st.rerun()
        else:
            st.error("Please use a valid Gmail address")

# Main App
def app_main():
    st.sidebar.write(f"Logged in as: `{st.session_state.user}`")
    if st.sidebar.button("Logout"):
        del st.session_state.user
        st.rerun()

    st.title("Campaign Caller")

    # Call Start Form
    with st.form("call_form"):
        number = st.text_input("Customer Number (+E.164)", key="number_input")
        submitted = st.form_submit_button("Start Call")
    if submitted:
        payload = {
            "assistantId": ASSISTANT_ID,
            "phoneNumberId": PHONE_ID,
            "customer": {"number": number}
        }
        resp = requests.post("https://api.vapi.ai/call/phone", json=payload, headers=HEADERS)
        if resp.ok:
            data = resp.json()
            st.success("Call initiated!")
            st.json(data)
            save_log({
                "time": datetime.utcnow().isoformat(),
                "call_id": data.get("id"),
                "number": number
            })
        else:
            st.error("Error: Unable to start call.")
            st.code(f"Status Code: {resp.status_code}")
            st.code(f"Response: {resp.text}")
            st.json(payload)


    # Fetch Transcript Section
    st.markdown("---")
    st.subheader("Fetch Call Transcripts")

    call_id = st.text_input("Enter Call ID to fetch", key="transcript_call_id")
    if st.button("Get Transcript", key="get_transcript_button"):
        details = fetch_call_details(call_id)
        parsed = []
        if details and isinstance(details, dict):
            transcript = details.get("transcript")
            if transcript:
                st.markdown("### 📄 Full Transcript")
                st.text(transcript)

                # Sentiment Analysis
                st.subheader("😊 Sentiment Analysis")
                blob = TextBlob(transcript)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                st.write(f"**Polarity:** {polarity:.2f} (−1 = Negative, +1 = Positive)")
                st.write(f"**Subjectivity:** {subjectivity:.2f} (0 = Objective, 1 = Subjective)")

                # QA Evaluation
                st.subheader("📊 QA Evaluation via LLaMA 4")
                prompt = (
                    "You are a call QA evaluator.\n"
                    "Based on the transcript below, evaluate the following questions.\n"
                    "Respond as a JSON list with each item containing:\n"
                    "- question\n"
                    "- rating (1–5 or 'N/A')\n"
                    "- explanation\n\n"
                    "Rating Scale: 1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent, N/A = Not Applicable\n\n"
                    f"Transcript:\n{transcript}\n\n"
                    "Questions:\n"
                )
                for q in qa_questions:
                    prompt += f"- {q}\n"

                try:
                    completion = groq_client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "You are a strict QA evaluator. Respond only in valid JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2
                    )
                    response = completion.choices[0].message.content
                    parsed = json.loads(response)

                    ratings = []
                    labels = []

                    st.subheader("📝 QA Evaluation Results")
                    for i, item in enumerate(parsed):
                        st.markdown(f"**Q{i+1}:** {item['question']}")
                        st.write(f"**Rating:** {item['rating']}")
                        st.write(f"**Explanation:** {item['explanation']}")

                        if str(item["rating"]).isdigit():
                            score = int(item["rating"])
                            st.progress(score / 5)
                            ratings.append(score)
                            labels.append(item['question'][:50] + "...")
                        else:
                            st.markdown("**Rating Scale:** N/A")

                        st.markdown("---")

                    if ratings:
                        st.subheader("📈 Ratings Overview")
                        fig, ax = plt.subplots(figsize=(8, len(ratings) * 0.4))
                        ax.barh(labels, ratings)
                        ax.set_xlabel("Rating (1–5)")
                        ax.set_xlim(0, 5)
                        ax.invert_yaxis()
                        st.pyplot(fig)

                except Exception as e:
                    st.error("LLaMA evaluation failed.")
                    st.exception(e)

            else:
                st.info("No transcript available.")

            # Download QA Evaluation
            structured = details.get('analysis', {}).get('structuredData')
            if structured:
                df_call = pd.DataFrame([structured])
                df_qa = pd.DataFrame(parsed) if parsed else pd.DataFrame([{"question": "N/A", "rating": "N/A", "explanation": "No evaluation"}])
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_call.to_excel(writer, index=False, sheet_name='CallData')
                    df_qa.to_excel(writer, index=False, sheet_name='QA Evaluation')
                st.download_button(
                    label="📥 Download Full Analysis (CallData + QA)",
                    data=output.getvalue(),
                    file_name='call_analysis.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        else:
            st.error("Could not fetch details or invalid response.")

    # Call Logs
    st.markdown("---")
    st.subheader("📁 Saved Call Logs")
    logs = load_logs()
    if logs:
        st.dataframe(logs)
    else:
        st.info("No calls logged yet.")

if "user" not in st.session_state:
    login()
else:
    app_main()
