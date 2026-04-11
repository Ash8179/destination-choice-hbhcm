"""
E.4 – HTML Text Translation Pipeline

This script:
Translate HTML-embedded English text to Chinese using GoogleTranslator,
with retry logic, structure preservation, and failure logging.

Author: Zhang Wenyu
Date: 2026-03-15
"""

import pandas as pd
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import time

failed_texts = []

def translate_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = GoogleTranslator(source='en', target='zh-CN').translate(text)
            if result is None:
                return text
            return result
        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Error:", text[:30], e)
            if attempt < max_retries - 1:
                print(f"Wait for 10s...")
                time.sleep(10)
            else:
                failed_texts.append(text)
                return text

def translate_html(html_text):
    if pd.isna(html_text) or html_text.strip() == "":
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for elem in soup.find_all(string=True):
        text = elem.strip()
        if text:
            translated = translate_with_retry(text)
            if translated:
                elem.replace_with(translated)
    return str(soup)

def main():
    df = pd.read_csv("/Users/zhangwenyu/Downloads/ALL.csv")
    translated_list = []

    for i, row in df.iterrows():
        zh = translate_html(row['EN'])
        translated_list.append(zh)
        time.sleep(0.5)
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(df)}")

    df['ZH-S'] = translated_list
    df.to_csv("/Users/zhangwenyu/Downloads/translated.csv", index=False)

    if failed_texts:
        print("\n=== Failed Texts ===")
        for t in set(failed_texts):
            print(t)
        pd.DataFrame({"failed": list(set(failed_texts))}).to_csv("/Users/zhangwenyu/Downloads/failed_texts.csv", index=False)
    else:
        print("\nSuccessful)

if __name__ == "__main__":
    main()
