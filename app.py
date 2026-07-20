import os
import re
import gradio as gr
import pdfplumber
from google import genai
from gtts import gTTS

# ==========================
# Gemini API Key
# ==========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash"


# ==========================
# Clean Text
# ==========================
def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'[#*_`>-]+', '', text)
    text = re.sub(r'[•●▪■◆]+', '', text)

    return text.strip()


# ==========================
# Extract Resume Text
# ==========================
def extract_resume(pdf_path):
    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# ==========================
# Resume Analyzer
# ==========================
def analyze_resume(pdf_file):
    try:
        if pdf_file is None:
            return "Please upload a resume."

        resume_text = extract_resume(pdf_file)

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
        return f"Error: {str(e)}"


# ==========================
# Skill Gap Analyzer
# ==========================
def skill_gap(company, skills):
    try:
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

    except Exception as e:
        return f"Error: {str(e)}"


# ==========================
# Interview Questions
# ==========================
def interview_questions(company):
    try:
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

    except Exception as e:
        return f"Error: {str(e)}"


# ==========================
# Mock Interview
# ==========================
INTERVIEWER_PROFILE = """
You are a Senior Technical Interviewer.

Evaluate the student's answer.
Provide feedback.
Ask the next interview question.
"""


def interview_chat(message):
    try:
        prompt = f"""
{INTERVIEWER_PROFILE}

Student Answer:
{message}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        text = clean_text(response.text)

        audio_file = "response.mp3"

        tts = gTTS(
            text=text,
            lang="en"
        )

        tts.save(audio_file)

        return text, audio_file

    except Exception as e:
        return f"Error: {str(e)}", None


# ==========================
# UI
# ==========================
with gr.Blocks(
    title="AI Powered Placement Assistant"
) as demo:

    gr.Markdown(
        "# 🚀 AI Powered Placement Assistant"
    )

    # Resume Analyzer
    with gr.Tab("Resume Analyzer"):

        resume = gr.File(
            label="Upload Resume PDF",
            file_types=[".pdf"],
            type="filepath"
        )

        output1 = gr.Textbox(
            label="Analysis",
            lines=15
        )

        btn1 = gr.Button(
            "Analyze Resume"
        )

        btn1.click(
            fn=analyze_resume,
            inputs=resume,
            outputs=output1
        )

    # Skill Gap Analyzer
    with gr.Tab("Skill Gap Analyzer"):

        company = gr.Textbox(
            label="Target Company"
        )

        skills = gr.Textbox(
            label="Your Skills"
        )

        output2 = gr.Textbox(
            label="Result",
            lines=15
        )

        btn2 = gr.Button(
            "Analyze Skills"
        )

        btn2.click(
            fn=skill_gap,
            inputs=[company, skills],
            outputs=output2
        )

    # Interview Questions
    with gr.Tab("Interview Questions"):

        company2 = gr.Textbox(
            label="Company Name"
        )

        output3 = gr.Textbox(
            label="Questions",
            lines=15
        )

        btn3 = gr.Button(
            "Generate Questions"
        )

        btn3.click(
            fn=interview_questions,
            inputs=company2,
            outputs=output3
        )

    # Mock Interview
    with gr.Tab("Mock Interview"):

        msg = gr.Textbox(
            label="Your Answer",
            lines=5,
            placeholder="Type your answer..."
        )

        text_output = gr.Textbox(
            label="AI Feedback",
            lines=10
        )

        audio_output = gr.Audio(
            label="Voice Feedback",
            type="filepath"
        )

        btn4 = gr.Button(
            "Submit Answer"
        )

        btn4.click(
            fn=interview_chat,
            inputs=msg,
            outputs=[
                text_output,
                audio_output
            ]
        )


# ==========================
# Launch
# ==========================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get(
                "PORT",
                10000
            )
        )
    )


