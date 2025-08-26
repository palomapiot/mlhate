from googletrans import Translator
import pandas as pd
import tqdm
import time
import asyncio

translator = Translator()

input_file = 'data/MetaHate<input_lang>.tsv'
output_file ='data/MetaHate<output_lang>.tsv'

df = pd.read_csv(input_file, sep='\t')
df.reset_index(drop=True, inplace=True)
df['id'] = df.index

try:
    df_parcial = pd.read_csv(output_file, sep='\t')
    df_parcial['id'] = range(len(df))
    translated_ids = df_parcial[df_parcial['text_<lang>'].notna() & (df_parcial['text_<lang>'] != '')]['id']
    print(f"{len(translated_ids)} messages already translated.")
except FileNotFoundError:
    df_parcial = df.copy()
    df_parcial['text_<lang>'] = None
    translated_ids = set()

batch_size = 50

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

for i in tqdm.tqdm(range(len(df)), desc="Translating..."):
    if df.loc[i, 'id'] in translated_ids:
        continue

    try:
        translated = loop.run_until_complete(translator.translate(df.loc[i, 'text'], dest='gl'))
        text_lang = translated.text
    except Exception as e:
        print(f"Error translating message with ID {df.loc[i, 'id']}: {e}")
        text_lang = None

    df_parcial.loc[i, 'text_<lang>'] = text_lang

    if (i + 1) % batch_size == 0 or i == len(df) - 1:
        df_parcial.to_csv(output_file, sep='\t', index=False)
        print(f"Saving parcial data at {output_file} (message {i + 1}/{len(df)}).")
        time.sleep(1)
