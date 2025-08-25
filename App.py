import streamlit as st
from streamlit_tags import st_tags
import pandas as pd
from PIL import Image
import io, re, random, time, datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile

# -------- Save AI-enhanced resume as PDF --------
def save_text_as_pdf(text: str, filename: str = "enhanced_resume.pdf") -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_file.name, pagesize=letter)
    width, height = letter
    y = height - 50
    for line in text.split("\n"):
        c.drawString(50, y, line.strip())
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    return temp_file.name

# -------- PDF parsing --------
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter

# Hugging Face imports
from huggingface_hub import InferenceClient

# -------- ATS Score Helper --------
def ats_score(resume_text, job_description):
    """Compute ATS-like score based on job description keywords matching."""
    job_keywords = set(re.findall(r'\w+', job_description.lower()))
    resume_keywords = set(re.findall(r'\w+', resume_text.lower()))
    matched_keywords = resume_keywords & job_keywords
    score = int(len(matched_keywords) / len(job_keywords) * 100) if job_keywords else 0
    return score, matched_keywords

# -------- PDF Text Extractor --------
def extract_text_from_pdf(uploaded_file) -> str:
    resource_manager = PDFResourceManager()
    retstr = io.StringIO()
    device = TextConverter(resource_manager, retstr, laparams=LAParams())
    interpreter = PDFPageInterpreter(resource_manager, device)
    uploaded_file.seek(0)
    for page in PDFPage.get_pages(uploaded_file, caching=True, check_extractable=True):
        interpreter.process_page(page)
    text = retstr.getvalue()
    device.close()
    retstr.close()
    return text

# -------- Regex Helpers --------
EMAIL_RE  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE  = re.compile(r"\+?\d[\d\s\-()]{8,}\d")

def extract_email(text:str) -> str:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else "Not found"

def extract_phone(text:str) -> str:
    m = PHONE_RE.search(text)
    return m.group(0) if m else "Not found"

# -------- Skills --------
SKILL_KEYWORDS = [
    "python","r","sql","numpy","pandas","matplotlib","seaborn","scikit-learn",
    "machine learning","deep learning","tensorflow","keras","pytorch","nlp",
    "data analysis","data visualization","statistics",
    "html","css","javascript","typescript","react","next.js","node","express","django","flask",
    "android","kotlin","java","swift","ios","flutter","dart",
    "git","linux","docker","kubernetes","aws","azure","gcp",
    "spark","hadoop","tableau","power bi","powerbi"
]

def extract_skills(text:str):
    text_low = text.lower()
    found = []
    for kw in SKILL_KEYWORDS:
        if kw in text_low:
            found.append(kw)
    return sorted(set(found), key=lambda x: found.index(x))

# -------- Course Picker --------
from Courses import (
    ds_course, web_course, android_course, ios_course, uiux_course,
    resume_videos, interview_videos
)

def pick_field_and_courses(skills:list):
    s = set(skills)
    if s & {"tensorflow","keras","pytorch","machine learning","deep learning","scikit-learn","python","pandas"}:
        return "Data Science / ML", ds_course
    if s & {"react","next.js","javascript","typescript","node","django","flask","html","css"}:
        return "Web Development", web_course
    if s & {"android","kotlin","java","flutter","dart"}:
        return "Android Development", android_course
    if s & {"swift","ios"}:
        return "iOS Development", ios_course
    if s & {"figma","ui","ux","adobe xd","power bi","tableau"}:
        return "UI/UX / Design", uiux_course
    return "General Upskilling", ds_course

def course_recommender(course_list, n=5):
    random.shuffle(course_list)
    return course_list[:n]

# -------- Hugging Face AI Resume Enhancement --------
# Replace YOUR_TOKEN_HERE with your valid Hugging Face token
HF_TOKEN = "hf_YOUR_TOKEN_HERE"
client = InferenceClient(token=HF_TOKEN)

def enhance_resume_with_hf(resume_text, role="general"):
    prompt = f"Rewrite this resume to make it professional, ATS-friendly, and tailored for a {role} role:\n\n{resume_text}"
    try:
        response = client.text_generation(
            model="google/flan-t5-large",
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True
        )
        if isinstance(response, dict) and "generated_text" in response:
            return response["generated_text"].strip()
        elif isinstance(response, str):
            return response.strip()
        else:
            return str(response)
    except Exception:
        return (
            "✅ Enhanced Resume (Fallback Mode)\n\n"
            "Hugging Face API token might be missing/invalid.\n\n"
            f"{resume_text.strip()}\n\n"
            "👉 Tips:\n"
            "- Use action verbs (Led, Designed, Built).\n"
            "- Add measurable achievements.\n"
            "- Tailor skills to job description.\n"
        )

# -------- Streamlit UI --------
st.set_page_config(page_title="Smart Resume Analyser", page_icon="📄")
st.title("📄 Resume Analyser And Enhancing (AI Powered)")

uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file:
    st.success("Resume uploaded ✅")
    with st.spinner("Extracting text..."):
        resume_text = extract_text_from_pdf(uploaded_file)

    st.subheader("📑 Extracted Text (preview)")
    st.write(resume_text[:1500] + (" ..." if len(resume_text) > 1500 else ""))

    # -------- Basic Info --------
    st.subheader("👤 Basic Info")
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    left, right = st.columns(2)
    left.write(f"**Email:** {email}")
    right.write(f"**Phone:** {phone}")

    # -------- Job Description for ATS --------
    st.subheader("📄 Job Description (Optional)")
    job_desc = st.text_area("Paste the job description here to get an ATS score", height=150)

    if job_desc:
        ats_score_value, matched_keywords = ats_score(resume_text, job_desc)
        st.subheader("📊 ATS Score")
        st.progress(ats_score_value)
        st.write(f"**ATS Score:** {ats_score_value}/100")
        if matched_keywords:
            st.write(f"Matched Keywords ({len(matched_keywords)}): {', '.join(list(matched_keywords)[:20])} ...")
        else:
            st.info("No keywords matched with the job description.")

    # -------- Skills --------
    st.subheader("🧩 Detected Skills")
    detected_skills = extract_skills(resume_text)
    if detected_skills:
        st_tags(label="Skills found in your resume:",
                text="(auto-detected — you can edit)",
                value=detected_skills, key="skills")
    else:
        st.info("No known skills auto-detected. You can still proceed.")

    # -------- Resume Section Checklist --------
    st.subheader("📝 Resume Content Checklist")
    checks = {
        "Objective / Summary":  any(w in resume_text.lower() for w in ["objective", "summary", "about me"]),
        "Projects":             "project" in resume_text.lower(),
        "Achievements/Awards":  any(w in resume_text.lower() for w in ["achievement","award","honor"]),
        "Hobbies/Interests":    any(w in resume_text.lower() for w in ["hobby","hobbies","interest","interests"]),
        "Declaration":          "declaration" in resume_text.lower(),
    }
    score = sum(20 for ok in checks.values() if ok)
    for label, ok in checks.items():
        st.write(("✅ " if ok else "❌ ") + label)

    st.subheader("📊 Resume Score")
    bar = st.progress(0)
    for i in range(score):
        time.sleep(0.01)
        bar.progress(i+1)
    st.success(f"Your Resume Score: **{score}/100**")

    # -------- Recommended Field & Courses --------
    st.subheader("🎯 Recommended Field & Courses")
    field, courses = pick_field_and_courses(detected_skills)
    st.write(f"**Suggested track:** {field}")
    recs = course_recommender(courses, n=5)
    for i, (name, link) in enumerate(recs, start=1):
        st.markdown(f"{i}. [{name}]({link})")

    # -------- AI Resume Enhancement --------
    if st.button("✨ Enhance Resume with AI (HuggingFace)"):
        with st.spinner("Enhancing your resume..."):
            enhanced_resume = enhance_resume_with_hf(resume_text, role=field)

        st.subheader("✨ AI-Enhanced Resume")
        st.write(enhanced_resume)

        # Convert to PDF
        pdf_path = save_text_as_pdf(enhanced_resume)
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download Enhanced Resume (PDF)",
                data=pdf_file,
                file_name="enhanced_resume.pdf",
                mime="application/pdf"
            )

    # -------- Videos --------
    st.subheader("🎥 Helpful Videos")
    st.write("Resume writing tips:")
    st.video(random.choice(resume_videos))
    st.write("Interview preparation:")
    st.video(random.choice(interview_videos))

    st.caption("Tip: tailor keywords in your resume to match the job description.")

else:
    st.info("Upload a PDF resume to begin.")