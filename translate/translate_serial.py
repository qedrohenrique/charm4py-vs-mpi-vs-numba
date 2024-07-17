import time

from deep_translator import GoogleTranslator


list_translated = []

f = open('../input', 'r')
input_text = f.read()
iterable_text = input_text.split('.')
f.close()

t0 = time.time()

print(f"Translating {len(iterable_text)} lines.\n")


for idx, line in enumerate(iterable_text):
    print(f"Current line: {idx} - {round(time.time() - t0, 3)}s")
    list_translated.append(GoogleTranslator(source='en', target='pt').translate(line))

print('\nElapsed time: ', round(time.time() - t0, 3), 's', sep='')

out_file = open('../out', 'w')
out_file.write(' '.join(list_translated))
