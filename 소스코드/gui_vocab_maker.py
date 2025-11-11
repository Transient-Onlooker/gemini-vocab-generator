#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import json
from pathlib import Path
from typing import List, Tuple
import requests
import queue
import sys
import random

# --- Configuration ---
def load_api_key():
    """Loads the Gemini API key from api.json."""
    try:
        # Look for api.json in the same directory as the script
        api_path = Path(__file__).parent / "api.json"
        if not api_path.exists():
            # If not found, check in the parent directory (for PyInstaller --onefile builds)
            api_path = Path(sys.executable).parent / "api.json"

        with open(api_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            key = config.get("GEMINI_API_KEY")
            if not key:
                raise KeyError
            return key
    except (FileNotFoundError, KeyError):
        messagebox.showerror(
            "API 키 오류",
            "api.json 파일을 찾을 수 없거나 파일 형식이 잘못되었습니다.\n"
            "실행 파일과 같은 폴더에 아래 형식으로 api.json 파일을 만들어주세요:\n\n"
            '{\n  "GEMINI_API_KEY": "실제_API_키_입력"\n}'
        )
        sys.exit(1)

GEMINI_API_KEY = load_api_key()

# --- Core Logic ---
def parse_vocab_block(vocab_block: str) -> List[Tuple[str, List[str]]]:
    pairs: List[Tuple[str, List[str]]] = []
    for raw in vocab_block.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("=", 1)
        if len(parts) < 2:
            continue
        word = parts[0].strip()
        meanings_raw = parts[1].strip()
        senses = [s.strip() for s in __import__("re").split(r"[;,]", meanings_raw) if s.strip()]
        if word and senses:
            pairs.append((word, senses))
    return pairs

def build_prompts(vocab_block: str, parsed: List[Tuple[str, List[str]]], question_type: str, num_sentences: int) -> tuple[str, str]:
    distribution_rule = "2. CRITICAL: The position of the correct answer MUST be randomized to ensure a balanced distribution. For the entire set of questions, each choice position (①, ②, ③, ④, ⑤) should be the correct answer at least 10% of the time. Avoid concentrating the answers on one or two numbers."
    self_correction_rule = "### Final Review\nBefore concluding your response, you MUST review the entire generated text one last time to ensure every single rule has been followed. Pay special attention that every question has exactly 5 numbered choices (① to ⑤). If you find any mistake, you must correct it before finishing."

    system_prompt = ""
    if question_type == "빈칸 추론":
        system_prompt_lines = [
            "You are an expert English vocabulary test maker for Korean students.",
            "Your task is to create multiple-choice questions that test understanding of words in context.",
            "Strictly follow all rules below.",
            "",
            "### Main Rule",
            "For each WORD and for each of its SENSEs, you must generate a complete question block.",
            "",
            "### Answer Generation Rules",
            "1. CRITICAL: DO NOT mark the correct answer in the choices. Instead, create a separate `[정답]` section at the very end of the entire output, listing each question number and its correct choice number.",
            distribution_rule,
            "",
            "### Output Structure (per question)",
            "1. Start with the question number (e.g., '1.').",
            "2. Add the title: '다음 빈칸에 공통으로 들어갈 말로 가장 적절한 것은?'",
            f"3. Provide exactly {num_sentences} distinct English sentences as context. Each sentence must have the word blanked out as '_______'.",
            "4. Provide exactly 5 answer choices (①, ②, ③, ④, ⑤).",
            "5. The choices must include one correct answer (the original WORD) and four plausible but incorrect distractors.",
            "6. Separate each full question block with a '---' line.",
            "",
            self_correction_rule,
        ]
        system_prompt = "\n".join(system_prompt_lines)
    elif question_type == "영영풀이":
        system_prompt_lines = [
            "You are an expert English vocabulary test maker for Korean students.",
            "Your task is to create multiple-choice questions based on English definitions.",
            "Strictly follow all rules below.",
            "",
            "### Main Rule",
            "For each WORD, you must generate one complete multiple-choice question.",
            "",
            "### Answer Generation Rules",
            "1. CRITICAL: DO NOT mark the correct answer in the choices. Instead, create a separate `[정답]` section at the very end of the entire output, listing each question number and its correct choice number.",
            distribution_rule,
            "",
            "### Output Structure (per question)",
            "1. Start with the question number (e.g., '1.').",
            "2. Add the title: '다음 영어 설명에 해당하는 단어는?'",
            "3. Provide the English definition of the WORD as the question body.",
            "4. Provide exactly 5 answer choices (①, ②, ③, ④, ⑤): one correct answer (the original WORD) and four plausible distractors (e.g., synonyms, related words).",
            "5. Separate each full question block with a '---' line.",
            "",
            self_correction_rule,
        ]
        system_prompt = "\n".join(system_prompt_lines)
    elif question_type == "뜻풀이 판단":
        system_prompt_lines = [
            "You are an expert English vocabulary test maker for Korean students.",
            "Your task is to create multiple-choice questions that test the precise definition of a word.",
            "Strictly follow all rules below.",
            "",
            "### Main Rule",
            "For each WORD, you must generate one complete multiple-choice question asking for its correct definition.",
            "",
            "### Answer Generation Rules",
            "1. CRITICAL: DO NOT mark the correct answer in the choices. Instead, create a separate `[정답]` section at the very end of the entire output, listing each question number and its correct choice number.",
            distribution_rule,
            "",
            "### Output Structure (per question)",
            "1. Start with the question number (e.g., '1.').",
            "2. Add the title: '다음 단어 <WORD>의 영영풀이로 가장 적절한 것은?' (replace <WORD> with the actual word).",
            "3. Provide exactly 5 definition choices (①, ②, ③, ④, ⑤): one perfectly correct definition and four subtly incorrect but plausible definitions.",
            "4. Separate each full question block with a '---' line.",
            "",
            self_correction_rule,
        ]
        system_prompt = "\n".join(system_prompt_lines)

    parsed_for_model_text = "\n".join([f"{word} = {', '.join(senses)}" for word, senses in parsed])
    user_prompt_lines = [
        "Here is the list of vocabulary. Create test questions based on these words, strictly following all rules defined in the system instructions.",
        "",
        "[Vocabulary List]",
        parsed_for_model_text
    ]
    user_prompt = "\n".join(user_prompt_lines)
    return system_prompt, user_prompt

def call_gemini(api_key: str, model: str, system_prompt: str, user_prompt: str, timeout: int = 120) -> str:
    assert api_key, "Gemini API 키가 비어 있습니다."
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{
            "role": "user",
            "parts": [{"text": user_prompt}],
        }],
        "generationConfig": {
            "temperature": 0.7
        },
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=timeout)
    if not resp.ok:
        raise RuntimeError(f"Gemini API 오류 {resp.status_code}: {resp.text}")
    payload = resp.json()
    try:
        text = payload["candidates"][0]["content"]["parts"][0]["text"].strip()
        if not text:
            raise KeyError("empty text")
        return text
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"응답 파싱 실패: {e}\n원본: {json.dumps(payload)[:1000]} ...")

# --- GUI Application ---
class VocabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("영어 단어 문제 생성기")
        self.root.geometry("850x600")
        self.input_filepath = None
        self.result_queue = queue.Queue()
        self.sentence_limit_unlocked = False

        self.model_map = {
            "Gemini 2.5 Pro (회당 ~5원)": "gemini-2.5-pro",
            "Gemini 2.5 Flash (권장됨) (회당 ~1.2원)": "gemini-2.5-flash",
            "Gemini 2.5 Flash Lite (권장됨) (회당 ~0.2원)": "gemini-2.5-flash-lite",
            "Gemini 2.0 Flash (회당 ~0.2원)": "gemini-2.0-flash",
            "Gemini 2.0 Flash Lite (회당 ~0.15원)": "gemini-2.0-flash-lite"
        }
        self.model_display_names = list(self.model_map.keys())

        main_frame = tk.Frame(root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame_1 = tk.Frame(main_frame)
        control_frame_1.pack(fill=tk.X, pady=(0, 5))
        control_frame_2 = tk.Frame(main_frame)
        control_frame_2.pack(fill=tk.X, pady=(0, 5))

        text_frame = tk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(1, weight=1)
        text_frame.rowconfigure(1, weight=1)

        self.btn_load = tk.Button(control_frame_1, text="TXT 불러오기", command=self.load_file)
        self.btn_load.pack(side=tk.LEFT, padx=(0, 5))
        self.status_label = tk.Label(control_frame_1, text="파일을 불러오세요.")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        self.lbl_model = tk.Label(control_frame_2, text="모델:")
        self.lbl_model.pack(side=tk.LEFT)
        self.combo_model = ttk.Combobox(control_frame_2, values=self.model_display_names, state="readonly", width=35) # Increased width
        self.combo_model.set("Gemini 2.5 Flash Lite (권장됨) (회당 ~0.2원)") # Set default with cost
        self.combo_model.pack(side=tk.LEFT, padx=(0, 5))

        self.lbl_q_type = tk.Label(control_frame_2, text="문제 유형:")
        self.lbl_q_type.pack(side=tk.LEFT)
        self.question_types = ["빈칸 추론", "영영풀이", "뜻풀이 판단"]
        self.combo_q_type = ttk.Combobox(control_frame_2, values=self.question_types, state="readonly", width=12)
        self.combo_q_type.set(self.question_types[0])
        self.combo_q_type.pack(side=tk.LEFT, padx=5)
        self.combo_q_type.bind("<<ComboboxSelected>>", self.on_q_type_change)

        self.sentence_count_frame = tk.Frame(control_frame_2)
        self.lbl_sentence_count = tk.Label(self.sentence_count_frame, text="예문 개수:")
        self.lbl_sentence_count.pack(side=tk.LEFT)
        self.spin_sentence_count = tk.Spinbox(self.sentence_count_frame, from_=1, to=10, width=5)
        self.spin_sentence_count.pack(side=tk.LEFT, padx=(0,5))

        self.btn_generate = tk.Button(control_frame_2, text="문제 생성", command=self.start_generation_thread)
        self.btn_generate.pack(side=tk.LEFT, padx=5)
        self.btn_save = tk.Button(control_frame_2, text="결과 저장", command=self.save_result, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        tk.Label(text_frame, text="입력 미리보기").grid(row=0, column=0, sticky="w", pady=(5,2))
        self.text_input = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.text_input.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        tk.Label(text_frame, text="생성된 문제").grid(row=0, column=1, sticky="w", pady=(5,2))
        self.text_output = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.text_output.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        
        self.root.after(100, self.process_queue)
        self.on_q_type_change(None)

    def on_q_type_change(self, event):
        if self.combo_q_type.get() == "빈칸 추론":
            self.sentence_count_frame.pack(side=tk.LEFT, before=self.btn_generate, padx=5)
        else:
            self.sentence_count_frame.pack_forget()

    def load_file(self):
        filepath = filedialog.askopenfilename(title="단어장 TXT 파일 선택", filetypes=(("텍스트 파일", "*.txt"), ("모든 파일", "*.*")))
        if not filepath:
            return
        self.input_filepath = Path(filepath)
        try:
            content = self.input_filepath.read_text(encoding="utf-8-sig")
            self.text_input.config(state=tk.NORMAL)
            self.text_input.delete(1.0, tk.END)
            self.text_input.insert(tk.END, content)
            self.text_input.config(state=tk.DISABLED)
            self.status_label.config(text=f"로드: {self.input_filepath.name}")
            self.btn_save.config(state=tk.DISABLED)
            self.text_output.config(state=tk.NORMAL)
            self.text_output.delete(1.0, tk.END)
            self.text_output.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("파일 읽기 오류", f"파일을 읽는 중 오류가 발생했습니다:\n{e}")

    def start_generation_thread(self):
        if not self.input_filepath:
            messagebox.showwarning("알림", "먼저 TXT 파일을 불러오세요.")
            return

        model_display_name_with_cost = self.combo_model.get()
        model_base_name = model_display_name_with_cost.split(' (회당')[0] # Extract base name for warning checks

        if model_base_name == "Gemini 2.5 Pro":
            if not messagebox.askyesno("모델 선택 경고", "2.5 pro는 고도의 논리적 추론을 위해 만들어진 모델입니다. 2.5 pro를 이용할 경우 과도한 요금이 과금될 수 있습니다. 계속하시겠습니까?"):
                return
        elif model_base_name in ["Gemini 2.0 Flash Lite", "Gemini 2.0 Flash"]:
            if not messagebox.askyesno("모델 선택 경고", "성능이 낮은 모델은 문제 생성에 오류가 걸리거나 생성 시간이 오래걸릴 수 있을 수 있습니다. 계속하시겠습니까?"):
                return

        question_type = self.combo_q_type.get()
        num_sentences = 1
        if question_type == "빈칸 추론":
            try:
                num_sentences = int(self.spin_sentence_count.get())
                if num_sentences > 5 and not self.sentence_limit_unlocked:
                    if messagebox.askyesno("예문 개수 경고", "예문을 5개 이상 생성하면 과도한 요금이 과금될 수 있습니다. 계속하시겠습니까?"):
                        self.sentence_limit_unlocked = True
                        self.spin_sentence_count.config(to=50)
                    else:
                        return
            except (ValueError, tk.TclError):
                num_sentences = 1

        self.set_ui_state(tk.DISABLED)
        self.status_label.config(text="생성 중...")
        vocab_block = self.text_input.get(1.0, tk.END)
        model_id = self.model_map[model_display_name_with_cost]
        thread = threading.Thread(target=self.run_generation, args=(vocab_block, question_type, num_sentences, model_id), daemon=True)
        thread.start()

    def run_generation(self, vocab_block, question_type, num_sentences, model_id):
        try:
            parsed = parse_vocab_block(vocab_block)
            if not parsed:
                raise ValueError("입력에서 유효한 'word = 뜻' 형식을 찾을 수 없습니다.")
            
            random.shuffle(parsed)

            system_prompt, user_prompt = build_prompts(vocab_block, parsed, question_type, num_sentences)
            output_text = call_gemini(api_key=GEMINI_API_KEY, model=model_id, system_prompt=system_prompt, user_prompt=user_prompt)
            self.result_queue.put(("success", output_text))
        except Exception as e:
            self.result_queue.put(("error", str(e)))

    def process_queue(self):
        try:
            status, data = self.result_queue.get_nowait()
            if status == "success":
                self.text_output.config(state=tk.NORMAL)
                self.text_output.delete(1.0, tk.END)
                self.text_output.insert(tk.END, data)
                self.text_output.config(state=tk.DISABLED)
                self.status_label.config(text="생성 완료!")
                self.btn_save.config(state=tk.NORMAL)
            elif status == "error":
                messagebox.showerror("생성 오류", data)
                self.status_label.config(text="오류 발생")
            self.set_ui_state(tk.NORMAL)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def save_result(self):
        content_to_save = self.text_output.get(1.0, tk.END).strip()
        if not content_to_save:
            messagebox.showwarning("알림", "저장할 내용이 없습니다.")
            return
        original_name = self.input_filepath.stem if self.input_filepath else "result"
        q_type_short = {"빈칸 추론": "빈칸", "영영풀이": "영영", "뜻풀이 판단": "뜻풀이"}.get(self.combo_q_type.get(), "문제")
        suggested_filename = f"{original_name}_{q_type_short}.txt"
        filepath = filedialog.asksaveasfilename(title="결과 저장", initialfile=suggested_filename, defaultextension=".txt", filetypes=(("텍스트 파일", "*.txt"), ("모든 파일", "*.*")))
        if not filepath:
            return
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content_to_save)
            self.status_label.config(text=f"저장 완료: {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("저장 오류", f"파일 저장 중 오류가 발생했습니다:\n{e}")

    def set_ui_state(self, state):
        self.btn_load.config(state=state)
        self.btn_generate.config(state=state)
        self.combo_model.config(state="readonly" if state == tk.NORMAL else tk.DISABLED)
        self.combo_q_type.config(state="readonly" if state == tk.NORMAL else tk.DISABLED)
        if self.combo_q_type.get() == "빈칸 추론":
            try:
                self.spin_sentence_count.config(state="normal" if state == tk.NORMAL else tk.DISABLED)
            except tk.TclError:
                pass
        if state == tk.NORMAL and self.text_output.get(1.0, tk.END).strip():
             self.btn_save.config(state=tk.NORMAL)
        else:
             self.btn_save.config(state=tk.DISABLED)

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        messagebox.showerror("패키지 필요", "'requests' 라이브러리가 설치되지 않았습니다.\n프로그램을 종료하고 터미널(cmd)에서 'pip install requests'를 실행해주세요.")
        sys.exit(1)
    root = tk.Tk()
    app = VocabApp(root)
    root.mainloop()
