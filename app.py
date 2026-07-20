import os
import re
import gradio as gr
import pdfplumber
from google import genai
from gtts import gTTS

# Render Environment Variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"


def clean_text(text):
    text = re.sub(r'[#*_`>-]+', '', text)
    text = re.sub(r'[•●▪■◆]+', '', text)
    return text.strip()


def extract_resume(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def analyze_resume(pdf_path):
    try:
        resume_text = extract_resume(pdf_path)

        prompt = f"""
        Analyze this resume and provide:

        1. Resume Score out of 100
        2. Strengths
        3. Weaknesses
        4. Missing Skills
        5. Improvement Suggestions

        Resume:
        {resume_text}
        """

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return clean_text(response.text)

    except Exception as e:
        return str(e)


def skill_gap(company, skills):
    prompt = f"""
    Student Skills:
    {skills}

    Target Company:
    {company}

    Provide:
    1. Missing Skills
    2. Learning Roadmap
    3. Placement Preparation Tips
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return clean_text(response.text)


def interview_questions(company):
    prompt = f"""
    Generate interview questions for freshers applying to {company}.

    Include:
    - Technical Questions
    - HR Questions
    - Coding Questions
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return clean_text(response.text)


INTERVIEWER_PROFILE = """
You are a Senior Technical Interviewer.

Ask one interview question at a time.
Evaluate the student's answer.
Give feedback and ask the next question.
"""


def interview_chat(message, history):
    prompt = f"""
    {INTERVIEWER_PROFILE}

    Previous Chat:
    {history}

    Student Answer:
    {message}
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    text = clean_text(response.text)

    tts = gTTS(text=text, lang="en")
    audio_file = "response.mp3"
    tts.save(audio_file)

    return text, audio_file


with gr.Blocks() as demo:

    gr.Markdown("# AI Powered Placement Assistant")

    with gr.Tab("Resume Analyzer"):
        resume = gr.File(label="Upload Resume PDF")
        output1 = gr.Textbox(lines=15)
        btn1 = gr.Button("Analyze Resume")
        btn1.click(analyze_resume, inputs=resume, outputs=output1)

    with gr.Tab("Skill Gap Analyzer"):
        company = gr.Textbox(label="Target Company")
        skills = gr.Textbox(label="Your Skills")
        output2 = gr.Textbox(lines=15)
        btn2 = gr.Button("Analyze Skills")
        btn2.click(skill_gap,
                   inputs=[company, skills],
                   outputs=output2)

    with gr.Tab("Interview Questions"):
        company2 = gr.Textbox(label="Company")
        output3 = gr.Textbox(lines=15)
        btn3 = gr.Button("Generate Questions")
        btn3.click(interview_questions,
                   inputs=company2,
                   outputs=output3)

    with gr.Tab("Mock Interview"):
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Your Answer")
        text_output = gr.Textbox(label="Feedback")
        audio_output = gr.Audio(label="Voice Feedback")
        btn4 = gr.Button("Submit")

        btn4.click(
            interview_chat,
            inputs=[msg, chatbot],
            outputs=[text_output, audio_output]
        )


demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)