import os
import gradio as gr
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline

# 1. Watsonx credentials & Granite LLM initialization
project_id = os.environ.get("WATSONX_PROJECT_ID", "skills-network")
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=os.environ.get("WATSONX_API_KEY")
)

parameters = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MAX_NEW_TOKENS: 650,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.3,  # Lower temperature for higher factual precision
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
}

granite_llm = WatsonxLLM(
    model_id="ibm/granite-3-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=project_id,
    params=parameters
)

# 2. Preprocessing & Cybersecurity Terminology Cleaner
def remove_non_ascii(text: str) -> str:
    return "".join(i for i in text if ord(i) < 128)

def cyber_triage_assistant(ascii_transcript: str) -> str:
    """
    Cleans audio transcription errors common in security calls,
    standardizing CVEs, MITRE ATT&CK techniques, network protocols, and security tooling.
    """
    system_prompt = (
        "You are an expert cybersecurity triage specialist. Your task is to process transcripts "
        "of SOC incident bridge calls and security standups. Correct common speech-to-text audio "
        "transcription errors for technical security terms. Standardize naming conventions:\n"
        "- Ensure CVE IDs are properly capitalized and formatted (e.g., 'CVE-2024-XXXX').\n"
        "- Correct security tooling and concept names (e.g., 'EDR', 'SIEM', 'SOAR', 'Cobalt Strike', 'Mimikatz', 'Kerberoasting').\n"
        "- Fix network protocols, ports, and terms (e.g., 'RDP (Port 3389)', 'SMB (Port 445)', 'SSH (Port 22)').\n"
        "- Spell out critical acronyms with standard notation where appropriate (e.g., 'Indicator of Compromise (IoC)').\n"
        "Return the corrected transcript preserving all technical context."
    )
    
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\nTranscript:\n{ascii_transcript}"}
    ]
    
    llama_params = TextChatParameters(temperature=0.2, top_p=0.6)
    llama_model = ModelInference(
        model_id="meta-llama/llama-3-2-11b-vision-instruct",
        credentials=credentials,
        project_id=project_id,
        params=llama_params
    )
    
    response = llama_model.chat(messages=messages)
    return response["choices"][0]["message"]["content"]

# 3. LangChain Prompt Template for Incident Summarization
template = """You are a senior Incident Commander and SOC Lead. Generate a structured Security Incident Briefing and Action Plan based on the provided bridge call transcript.

Context Transcript:
{context}

Format your output exactly as follows:

### 1. Incident Executive Summary & Threat Scope
- Brief overview of the incident or operational status
- Affected infrastructure/assets and suspected threat vectors

### 2. Identified Indicators of Compromise (IoCs) & TTPs
- Observed IPs, hashes, domains, or compromised accounts
- Associated MITRE ATT&CK techniques or tools mentioned

### 3. Containment & Remediation Action Items
- Priority 1 (Immediate Containment): [Assignee / Action / Deadline]
- Priority 2 (Forensic Investigation & Eradication): [Assignee / Action / Deadline]
- Priority 3 (Hardening & Post-Incident Review): [Assignee / Action / Deadline]
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": RunnablePassthrough()}
    | prompt
    | granite_llm
    | StrOutputParser()
)

# 4. Speech-to-Text Pipeline using Whisper
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    chunk_length_s=30
)

def process_security_briefing(audio_file):
    if not audio_file:
        return "No audio file provided.", None

    # Step A: Transcribe audio to text
    raw_transcript = asr_pipeline(audio_file, batch_size=8)["text"]
    
    # Step B: Sanitize non-ASCII and clean cybersecurity terminology
    clean_ascii = remove_non_ascii(raw_transcript)
    standardized_transcript = cyber_triage_assistant(clean_ascii)
    
    # Step C: Generate structured incident briefing
    briefing = chain.invoke(standardized_transcript)
    
    # Step D: Save output report
    output_file = "incident_response_action_plan.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(briefing)
        
    return briefing, output_file

# 5. Gradio SOC Assistant Interface
audio_input = gr.Audio(sources="upload", type="filepath", label="Upload Incident Bridge Recording (.wav / .mp3)")
output_text = gr.Textbox(label="Incident Briefing & Containment Plan", lines=18)
download_file = gr.File(label="Download Incident Action Plan (.txt)")

iface = gr.Interface(
    fn=process_security_briefing,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="SOC Bridge & Incident Response Assistant",
    description="Upload an audio recording of an Incident Response bridge call or SOC standup. The assistant transcribes the conversation, standardizes security IoCs/TTPs, and generates a structured incident response action plan.",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=5000)