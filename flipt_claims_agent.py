#Installation
!pip install -q ipywidgets sentence-transformers google-genai numpy
# ---Imports & setup ---
import os
import json
import uuid
import re
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer
from google import genai
from google.colab import drive
# ---Mount Drive & configure credentials ---
drive.mount('/content/drive', force_remount=True)
SERVICE_ACCOUNT_PATH = "/content/drive/My Drive/my_service_account.json" #
Replace with your Service account details
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
os.environ["GOOGLE_CLOUD_PROJECT"] = "my_project_id" # Replace with your project id.
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
# ---Initialize Vertex AI / GenAI client ---
client = genai.Client(
 vertexai=True,
 project=os.environ["GOOGLE_CLOUD_PROJECT"],
 location=os.environ["GOOGLE_CLOUD_LOCATION"]
)
# ---Load claims data ---
CLAIMS_JSON_PATH = "/content/drive/My Drive/claims.json"
with open(CLAIMS_JSON_PATH, "r") as f:
 claims_data = json.load(f)
# ---Utility functions ---
def _sanitize(x):
 if x is None:
 return ""
 if isinstance(x, (bool, int, float)):
 return str(x)
 if isinstance(x, str):
 return x.strip()
 if isinstance(x, (bytes, bytearray)):
 return x.decode("utf-8", errors="ignore")
 return str(x)
def metadata_matches_free_text(md: dict, user_input: str) -> bool:
 ui = user_input.lower()
 drug = md.get("drug_name","").lower()
 if drug and drug in ui:
 return True
 pharm = md.get("pharmacy_name","").lower()
 if pharm and pharm in ui:
 return True
 # optionally match date if user writes a date string
 dt = md.get("paid_date","").lower()
 if dt and re.search(re.escape(dt), ui):
 return True
 return False
# --- Vectorization: Prepare embeddings + metadata store ---
embedder = SentenceTransformer('all-MiniLM-L6-v2')
claim_embeddings = []
claim_metadata = []
for raw in claims_data:
 info = raw.get("claim", {})
 resp = info.get("claim_response", {})
 claim_id = _sanitize(info.get("claim_tracking_id","")) or str(uuid.uuid4())
 status_raw = _sanitize(resp.get("transaction_response_status","")).upper()
 drug = _sanitize(info.get("drug_name",""))
 pharmacy = _sanitize(info.get("pharmacy_name",""))
 paid_amount = _sanitize(resp.get("total_amount_paid") or
resp.get("patient_pay_amount",""))
 paid_date = _sanitize(resp.get("date_of_service",""))
 reject_code = _sanitize(resp.get("reject_code",""))
 reject_message = _sanitize(resp.get("message",""))
 orig_ref = _sanitize(info.get("claim_request",
{}).get("associated_prescription_reference_number",""))
 md = {
 "claim_id": claim_id,
 "status_raw": status_raw,
 "drug_name": drug,
 "pharmacy_name": pharmacy,
 "paid_amount": paid_amount,
 "paid_date": paid_date,
 "reject_code": reject_code,
 "reject_message": reject_message,
 "original_claim_ref": orig_ref
 }
 claim_metadata.append(md)
 desc = f"{drug} {pharmacy} {paid_date} {claim_id}"
 emb = embedder.encode(desc)
 claim_embeddings.append(emb)
claim_embeddings = np.array(claim_embeddings)
# ---Main function with RAG layer---
def analyze_input(user_input: str) -> str:
 ui = _sanitize(user_input)
 if not ui:
 return "Submit your question here."
 # 1. Metadata-based filtering
 filtered = [md for md in claim_metadata if metadata_matches_free_text(md, ui)]
 # 2. Semantic embedding-based fallback
 ui_emb = embedder.encode([ui])[0]
 sims = np.dot(claim_embeddings, ui_emb)
 top_idxs = sims.argsort()[-5:][::-1]
 semantic_candidates = [claim_metadata[i] for i in top_idxs]
 # 3. Merge unique candidates
 candidate_map = { md["claim_id"]: md for md in filtered + semantic_candidates }
 candidates = list(candidate_map.values())
 # 4. Structured metadata & operational instructions
 prompt = "You are a pharmacy-claims AI Agent. Use ONLY the data below — do NOT
invent facts or hallucinate.\n\n"
 prompt += "=== CLAIMS CONTEXT ===\n"
 for c in candidates:
 prompt += f"claim_id: {c.get('claim_id','')}\n"
 prompt += f"status: {c.get('status_raw','')}\n"
 if c.get("drug_name"):
 prompt += f"drug_name: {c.get('drug_name')}\n"
 if c.get("pharmacy_name"):
 prompt += f"pharmacy_name: {c.get('pharmacy_name')}\n"
 if c.get("paid_amount"):
 prompt += f"paid_amount: {c.get('paid_amount')}\n"
 if c.get("paid_date"):
 prompt += f"paid_date: {c.get('paid_date')}\n"
 if c.get("reject_code"):
 prompt += f"reject_code: {c.get('reject_code')}\n"
 if c.get("reject_message"):
 prompt += f"reject_message: {c.get('reject_message')}\n"
 if c.get("original_claim_ref"):
 prompt += f"original_claim_ref: {c.get('original_claim_ref')}\n"
 prompt += "---\n"
 prompt += f"\nUser query:\n\"\"\"{ui}\"\"\"\n\n"
 prompt += (
 "Instructions:\n"
 "- Based only on the data above, decide whether the user refers to a Paid, Rejected, or
Duplicate claim.\n"
 "- Always include the Claim ID in your answer.\n"
 "- If Paid: return payment info (paid amount, date of service/paid_date,
payee/pharmacy, claim ID).\n"
 "- If Rejected: return reject code and message, and suggest how to fix or resubmit if
possible.\n"
 "- If Duplicate: if original_claim_ref is present, return that as original; otherwise say you
don’t have enough information to find original claim.\n"
 "- If none of the above or data is insufficient: say \"I don’t have enough information.\"\n"
 "Use plain English, short sentences, no jargon.\n"
 )
 response = client.models.generate_content(
 model="gemini-3-pro-preview",
 contents=prompt
 )
 return _sanitize(response.text)
#---Chat Interface---
import ipywidgets as widgets
from IPython.display import display, HTML
# Title for the chat interface
title = widgets.HTML(value="<h2>FliptRx AI Lead Mentor</h2>")
# Input field for claim description or question with adjusted size
input_box = widgets.Textarea(
 value="",
 placeholder="What can I assist you with today?",
 description="Input:",
 layout=widgets.Layout(width='20%', height='400px')
)
# Button to trigger analysis
button = widgets.Button(description="Ask")
# Output widget for displaying the chat messages
output = widgets.Output()
# When the button is clicked, analyze input
def on_click(_):
 with output:
 output.clear_output()
 text = input_box.value.strip()
 if not text:
 print("Please type your question here.")
 return
 try:
 resp = analyze_input(text)
 print(resp)
 except Exception as e:
 print("Error:", e)
button.on_click(on_click)
# Create the layout with title, input box, button, and output container
ui_layout = widgets.VBox([title, input_box, button, output])