# 🎓 Unified University AI Chatbot System

An intelligent, multi-domain AI chatbot designed to assist students with:

- 🎓 Admissions queries
- 📚 Course & syllabus understanding
- 🌐 Website navigation
- 💼 Placement analytics

---

## 📸 UI Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/f1645a95-0a9a-47ac-8069-9bd4dbea102c" width="45%" />
  <img src="https://github.com/user-attachments/assets/9449b403-84cc-4ee8-a9ff-8472d3db35d4" width="45%" />
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/44d75a0b-d062-41aa-a2a7-ebe96adaba14" width="45%" />
  <img src="https://github.com/user-attachments/assets/866d7020-ca13-419b-8d35-f349f2a0d1ad" width="45%" />
</p>

---

Built using **RAG (Retrieval-Augmented Generation)**, **LLMs**, and real-world datasets.

---

## 🚀 Live Demo

👉 https://university-chatbot-ba8t.onrender.com/

---

## 🧠 Features

- 🔹 Multi-domain chatbot (Admissions, Courses, Navigation, Placements)
- 🔹 RAG-based responses using ChromaDB
- 🔹 Streaming responses (real-time)
- 🔹 Smart follow-up question suggestions
- 🔹 Placement insights using Pandas + LLM
- 🔹 Modern UI with dark/light mode

---

## 🏗️ Architecture

```text
Frontend (HTML + Tailwind)
        ↓
FastAPI Backend
        ↓
-----------------------------------
| Admissions RAG (ChromaDB)       |
| Course RAG (Dept-wise)          |
| Placement Agent (Pandas + LLM)  |
-----------------------------------
        ↓
Google Gemini API
```

---

## 🧩 Tech Stack

**Backend**
- FastAPI
- LangChain
- ChromaDB
- SentenceTransformers
- Pandas

**Frontend**
- HTML
- Tailwind CSS
- JavaScript

**AI/ML**
- Google Gemini API
- RAG
- Embeddings (MiniLM)

---

## 📂 Project Structure

```
.
├── integrated_main.py      # Main FastAPI app
├── app.py                  # Admissions chatbot
├── chatbot_script.py       # Course chatbot
├── placements_chatbot.py   # Placement agent
├── integrated_chat.html    # Frontend UI
├── data/                   # Course data
├── chroma_db/              # Vector DB
├── placements_data.csv     # Dataset
└── .env                    # Environment variables
```

---

## ⚙️ Setup

**1. Clone Repo**
```bash
git clone https://github.com/your-username/university-chatbot.git
cd university-chatbot
```

**2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Add `.env`**
```env
GOOGLE_API_KEY=your_key
GEMINI_API_KEY=your_key
```

**5. Run Server**
```bash
uvicorn integrated_main:app --reload
```

**6. Open Frontend**

Open `integrated_chat.html`

---

## 🧪 Example Queries

**🎓 Admissions**
- Eligibility for B.Tech?
- Scholarship details?

**📚 Courses**
- Syllabus for Data Structures
- Courses in semester 3

**💼 Placements**
- Highest package?
- Students placed in TCS?

---

## 🔥 Highlights

- ✔ Multi-agent AI system
- ✔ RAG + Data Agent integration
- ✔ Handles structured & unstructured data
- ✔ Low-latency streaming responses
- ✔ Scalable backend architecture

---

## 🧠 Challenges Solved

- Lazy loading to avoid startup delays
- Streaming LLM responses in FastAPI
- Reducing hallucination using RAG
- Integrating multiple AI pipelines

---

## 📈 Future Improvements

- Voice chatbot
- Multi-language support
- Fine-tuned models
- User authentication

---

## 👨‍💻 Author

**Chaitanya Mulampaka**  
Computer Science Engineering Student | AI & Data Science

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
