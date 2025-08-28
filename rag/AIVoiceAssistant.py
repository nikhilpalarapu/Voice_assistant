from __future__ import annotations
import re
from typing import Optional, Tuple
from datetime import datetime
import csv
import os

from qdrant_client import QdrantClient

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import ServiceContext

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import warnings
warnings.filterwarnings("ignore")


# -----------------------
# Helpers: parsing / fuzzy
# -----------------------

_STOP_WORD_NAMES = {"hi", "hello", "hey", "hiya", "yo", "hii"}

def _extract_name(text: str) -> Optional[str]:
    """Extract a reasonable one-word name. Avoids greeting words."""
    t = (text or "").strip()
    if not t:
        return None

    # Common patterns: "my name is X", "i am X", "i'm X"
    m = re.search(r"\bmy\s+name\s+is\s+([A-Za-z][A-Za-z\-']{1,30})\b", t, re.I)
    if not m:
        m = re.search(r"\bi\s*am\s+([A-Za-z][A-Za-z\-']{1,30})\b", t, re.I)
    if not m:
        m = re.search(r"\bi['’]m\s+([A-Za-z][A-Za-z\-']{1,30})\b", t, re.I)
    if m:
        cand = m.group(1).strip().lower()
        if cand not in _STOP_WORD_NAMES:
            return cand.title()

    # Fallback: single token that isn't just a greeting
    words = re.findall(r"[A-Za-z][A-Za-z\-']{1,30}", t)
    if len(words) == 1:
        cand = words[0].lower()
        if cand not in _STOP_WORD_NAMES:
            return words[0].title()

    return None


# all variants we frequently see when Whisper struggles
_BURGER_HINTS = {
    "burger","bugger","burder","burga","burjer","barger","bogger","booger","brgr","bgr","bug",
}
_PIZZA_HINTS = {
    "pizza","piza","pisa","pissa","pica","pizaah","pitha","pizzah","pizaaa",
    "piece","peace","peetsuh","peetsa",
}

def _normalize_tokens(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def _extract_item(text: str) -> Optional[str]:
    """
    Robustly map the utterance to 'burger' or 'pizza'.
    Accepts:
      - fuzzy variants (bugger/badger -> burger, piza/peace -> pizza)
      - digits: 1 -> burger, 2 -> pizza
      - code words: 'red' -> burger, 'green' -> pizza
    """
    t = (text or "").lower().strip()
    if not t:
        return None

    # digits
    if re.search(r"\b1\b", t):
        return "burger"
    if re.search(r"\b2\b", t):
        return "pizza"

    # code words (short, clear color words — optional, just to help speech)
    toks = _normalize_tokens(t)
    if "red" in toks:
        return "burger"
    if "green" in toks:
        return "pizza"

    # explicit ordinary words
    if "burger" in toks:
        return "burger"
    if "pizza" in toks:
        return "pizza"

    # fuzzy-ish: check for common mishears
    if toks & _BURGER_HINTS:
        return "burger"
    if toks & _PIZZA_HINTS:
        return "pizza"

    # last resort: crude similarity (SequenceMatcher) to target words
    try:
        from difflib import SequenceMatcher
        def sim(a, b): return SequenceMatcher(None, a, b).ratio()
        best: Tuple[str, float] = ("", 0.0)
        for w in toks:
            for target in ("burger", "pizza"):
                s = sim(w, target)
                if s > best[1]:
                    best = (target, s)
        if best[1] >= 0.75:
            return best[0]
    except Exception:
        pass

    return None


def _extract_yes_no(text: str) -> Optional[bool]:
    t = (text or "").lower()
    if re.search(r"\b(yes|yeah|yep|sure|ok|okay|please|confirm|do it)\b", t):
        return True
    if re.search(r"\b(no|nope|cancel|not now|later|stop)\b", t):
        return False
    # digits: 1=yes, 2=no are not used here (reserved for item)
    return None


# -----------------------
# Assistant with state + order log
# -----------------------

class AIVoiceAssistant:
    """
    Simple voice-friendly flow (name -> item -> price -> confirm -> end).
    Extremely forgiving recognition for burger/pizza.
    Ends cleanly after 'no', and resets for next customer.
    Also logs confirmed orders to orders.csv and keeps an in-memory list.
    """

    def __init__(self):
        # ---- Qdrant / RAG (kept for KB Q&A if you expand later) ----
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)

        self._llm = Ollama(
            model="mistral",
            base_url="http://127.0.0.1:11434",
            is_chat=False,
            request_timeout=120.0,
            temperature=0.2,
            additional_kwargs={
                "options": {
                    "num_gpu": 0,       # CPU for stability
                    "num_predict": 40,  # short answers when LLM is used
                }
            },
        )

        self._embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        self._service_context = ServiceContext.from_defaults(
            llm=self._llm,
            embed_model=self._embed,
        )

        self._index = None
        self._create_kb()
        self._create_chat_engine()

        # ---- Dialog state ----
        self._reset_session()

        # minimal menu (prices in ₹)
        self._menu = {"burger": 120, "pizza": 180}

        # ---- Order logging ----
        self._orders = []                   # in-memory for current run
        self._orders_csv = "orders.csv"     # persisted across runs
        self._ensure_csv_header()

    # ---------- KB / RAG ----------
    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(input_files=["rag/restaurant_file.txt"])
            documents = reader.load_data()

            vector_store = QdrantVectorStore(
                client=self._client,
                collection_name="kitchen_db",
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            self._index = VectorStoreIndex.from_documents(
                documents,
                service_context=self._service_context,
                storage_context=storage_context,
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        # Only used if you later want to answer KB questions in free-form
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt="Answer concisely (<= 25 words).",
        )

    # ---------- Session control ----------
    def _reset_session(self):
        self._state = "ASK_NAME"   # ASK_NAME -> ASK_ITEM -> QUOTE_PRICE -> CONFIRM/DONE
        self._customer_name: Optional[str] = None
        self._item: Optional[str] = None

    # public reset so app.py can call safely
    def reset_session(self):
        self._reset_session()

    # ---------- Orders CSV ----------
    def _ensure_csv_header(self):
        if not os.path.exists(self._orders_csv):
            with open(self._orders_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "name", "item", "price"])

    def _record_order(self, name: str, item: str, price: int):
        ts = datetime.now().isoformat(timespec="seconds")
        row = {"timestamp": ts, "name": name, "item": item, "price": price}
        self._orders.append(row)
        with open(self._orders_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts, name, item, price])

    def get_orders(self):
        """Return the in-memory list of orders for the current run."""
        return list(self._orders)

    def clear_orders(self):
        """Clear only in-memory orders (does not delete CSV)."""
        self._orders.clear()

    # ---------- Main flow ----------
    def interact_with_llm(self, customer_query: str) -> str:
        text = (customer_query or "").strip()

        # IDLE/DONE → if user says anything like "restart" or "order", start fresh
        if self._state == "DONE":
            yn = _extract_yes_no(text)
            # If they say 'no' again, just end gracefully
            if yn is False:
                return "Thanks for visiting Bangalore Kitchen. Goodbye!"
            # Any new utterance after done → start a new session
            self._reset_session()

        # 1) Ask / capture name
        if self._state == "ASK_NAME":
            name = _extract_name(text)
            if name:
                self._customer_name = name
                self._state = "ASK_ITEM"
                # Brief hint for low-quality audio (digits + color code are optional)
                return f"Hi {name}. Burger or pizza? You can also say '1' for burger or '2' for pizza."
            # If user just says "hi", still ask name (don’t move on)
            return "Welcome to Bangalore Kitchen. What is your name?"

        # 2) Ask / capture item
        if self._state == "ASK_ITEM":
            item = _extract_item(text)
            if item in self._menu:
                self._item = item
                self._state = "QUOTE_PRICE"
                price = self._menu[item]
                return f"{item.title()} is ₹{price}. Should I place the order?"
            # repeat, with the robust hint
            return "Burger or pizza? You can also say '1' for burger or '2' for pizza."

        # 3) Confirm
        if self._state == "QUOTE_PRICE":
            yn = _extract_yes_no(text)
            if yn is True:
                self._state = "DONE"
                name = self._customer_name or "Guest"
                item = self._item or "item"
                price = self._menu.get(item, 0)

                # record order to CSV + memory
                self._record_order(name, item, price)

                return f"Order placed for {item} under {name}. Total ₹{price}. Anything else?"
            if yn is False:
                # let them choose again
                self._state = "ASK_ITEM"
                return "No problem. Burger or pizza?"
            # unclear -> reprompt
            return "Please say yes or no."

        # 4) After done: handle “anything else”
        if self._state == "DONE":
            yn = _extract_yes_no(text)
            if yn is True:
                # start a new order immediately
                self._reset_session()
                return "Great. What is your name?"
            if yn is False:
                # end and reset
                self._reset_session()
                return "Thanks for visiting Bangalore Kitchen. Goodbye!"
            # still unclear: short reprompt
            return "Anything else? Say yes or no."

        # Shouldn’t happen
        return "Sorry, please repeat."
