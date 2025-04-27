# Nuggets_AI Gen AI Internship Assignment  
**Restaurant Data Scraper & RAG-based Chatbot**

## Overview  
This project enhances the Zomato experience by building a chatbot that can respond to natural language queries about restaurants, powered by the latest restaurant data collected through web scraping. The project is divided into two key components:

1. **Restaurant Data Scraper**: Collects data from various restaurant websites.
2. **Retrieval-Augmented Generation (RAG)-based Chatbot**: Utilizes this data to answer user queries.

## Problem Statement  
Consumers often seek specific information about restaurants that is not easily searchable, such as:
- "Which restaurant has the best vegetarian options?"
- "What are the gluten-free options at ABC Restaurant?"
- "Compare the spice levels at XYZ and ABC restaurants."
- "What’s the price range of desserts at ABC Restaurant?"

This project addresses these needs by:
1. Scraping restaurant data from multiple sources.
2. Processing and indexing this data.
3. Building a chatbot that retrieves relevant information and generates accurate responses to user queries.


## 📁 Project Structure

```bash
.
├── chatbot
│   ├── __init__.py
│   └── chat.py
├── data
│   ├── parsed
│   ├── processed
│   └── raw
├── docs
│   └── design.md
├── rag
│   ├── index.py
│   ├── retriever.py
│   ├── generator.py
│   └── utils.py
├── scraper
│   ├── scraper.py
│   └── parse.py
├── scripts
│   └── create_docs.py
├── requirements.txt
├── README.md
└── venv/    (optional after virtual environment setup)
```

### Description of Important Folders

- **chatbot/** — Contains the chatbot interface code.
- **data/** — Stores the raw, parsed, and processed restaurant data.
- **docs/** — Contains design documents related to system architecture.
- **rag/** — Contains the retrieval and generation logic used in the RAG architecture.
- **scraper/** — Contains the web scraper scripts for restaurant data extraction.
- **scripts/** — Script to process scraped data and create the knowledge base.
- **requirements.txt** — Lists all Python dependencies required to run the project.
- **README.md** — Project overview, setup instructions, and technical details.

---


## Components

### 1. **Web Scraper**  
The scraper extracts restaurant data, including:
- Name, location, and contact details
- Menu items, descriptions, and prices
- Features like vegetarian options, spice levels, etc.

Files:
- **scraper/scraper.py**: Handles the web scraping logic.
- **scraper/parse.py**: Processes the scraped data into a structured format.

### 2. **Knowledge Base**  
Once scraped, the data is cleaned and processed into a structured format. It’s then saved in JSON and HuggingFace dataset formats for use in the RAG model.

Files:
- **scripts/create_docs.py**: Converts raw scraped data into the final knowledge base.
- **data/parsed/**: Raw, unprocessed scraped data.
- **data/processed/**: Cleaned, structured data ready for use.
- **data/raw/**: Contains the unprocessed data.

### 3. **RAG-based Chatbot**  
This is the core of the project. It answers user queries by:
1. **Retrieving** relevant documents using Dense Passage Retrieval (DPR).
2. **Generating** responses using a pre-trained model based on the retrieved documents.

Files:
- **rag/index.py**: Main entry point for the RAG-based chatbot.
- **rag/retriever.py**: Fetches relevant documents.
- **rag/generator.py**: Generates answers based on retrieved documents.
- **rag/utils.py**: Utility functions for the retriever and generator.

### 4. **Chatbot Interface**  
The user can interact with the chatbot, which will answer queries about restaurant menus and features.

Files:
- **chatbot/chat.py**: Defines the chatbot logic.
- **chatbot/__init__.py**: Initializes the chatbot module.

## Setup & Installation

### 1. **Clone the Repository**
Clone the repository from GitHub:
```bash
git clone https://github.com/yourusername/zomato-gen-ai-internship.git
cd zomato-gen-ai-internship
```
##2. Create a Virtual Environment
Create a virtual environment for the project:
```bash
python -m venv venv
```


**Activate the environment:**
Windows:

```bash
.\venv\Scripts\activate
```
Linux/macOS:
```bash
source venv/bin/activate
```
**3. Install Dependencies**:
Install all the required dependencies:
```bash
pip install -r requirements.txt
```
**4. Run the Web Scraper
Execute the scraper to collect restaurant data:**
```bash
python scraper/scraper.py
```

**5. Generate Knowledge Base
After collecting data, process it and create the knowledge base:**
```bash
python scripts/create_docs.py
```
**6. Run the Chatbot
To interact with the chatbot, run the chatbot interface:**
```bash
python chatbot/chat.py
```

## Technical Details

### Architecture
This project follows the **Retrieval-Augmented Generation (RAG)** architecture.  
The retriever uses **Dense Passage Retrieval (DPR)** to fetch relevant information, while the generator uses a pre-trained model to produce natural language responses.

### Challenges Faced & Solutions

- **Data Cleaning**:  
  The scraped data was often inconsistent. We performed extensive cleaning and preprocessing to ensure the data was usable.

- **Efficient Indexing**:  
  We used **HuggingFace’s dataset format** for quick and scalable document retrieval.

- **Handling Complex Queries**:  
  We designed the chatbot to handle diverse user queries by combining information retrieval with generative models.

### Future Improvements

- **Scaling**:  
  The scraper can be extended to scrape more restaurant websites, and the data can be updated periodically.

- **Better Query Understanding**:  
  We can fine-tune the generative model to better handle complex or ambiguous queries.

- **Deployment**:  
  We could deploy the solution as a web-based application for real-time interactions.

### Deliverables

- **Code**: Complete source code for all components.
- **Scraped Data**: Structured restaurant data collected from multiple sources.
- **Documentation**: Detailed explanation of the system architecture, implementation, and challenges faced.
- **Demo Video**: A short video showcasing how the chatbot works with sample interactions.
## Demo Video

Watch the demo here: [Demo Video](https://youtube.com/shorts/tdckd8bvvGk?si=SSOnSRk-taI8p6uW)
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







