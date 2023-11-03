from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json

def extract_triplets_typed(text):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_, object_type, subject_type = '','','','',''

    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").replace("tp_XX", "").replace("__en__", "").split():
        if token == "<triplet>" or token == "<relation>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                relation = ''
            subject = ''
        elif token.startswith("<") and token.endswith(">"):
            if current == 't' or current == 'o':
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
                object_ = ''
                subject_type = token[1:-1]
            else:
                current = 'o'
                object_type = token[1:-1]
                relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '' and object_type != '' and subject_type != '':
        triplets.append({'head': subject.strip(), 'head_type': subject_type, 'type': relation.strip(),'tail': object_.strip(), 'tail_type': object_type})
    return triplets

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", src_lang="es_XX", tgt_lang="tp_XX") 
# Here we set English ("en_XX") as source language. To change the source language swap the first token of the input for your desired language or change to supported language. For catalan ("ca_XX") or greek ("el_EL") (not included in mBART pretraining) you need a workaround:
# tokenizer._src_lang = "ca_XX"
# tokenizer.cur_lang_code_id = tokenizer.convert_tokens_to_ids("ca_XX")
# tokenizer.set_src_lang_special_tokens("ca_XX")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large")
gen_kwargs = {
    "max_length": 256,
    "length_penalty": 1,
    "num_beams": 5,
    "num_return_sequences": 5,
    "forced_bos_token_id": None,
}

text = """
Sobre la mesa María, ve una manzana, un sombrero, una llave y un paraguas.
"""

text = """
Según el artículo 3, apartado 4, de la Directiva 2009/28/CE del Parlamento Europeo y del Consejo ( 3 ), cada
Estado miembro debe velar por que la cuota de energía procedente de fuentes renovables en todos los tipos de
transporte en 2020 sea como mínimo equivalente al 10 % del consumo final de energía en el transporte en dicho
Estado miembro.
"""
# Tokenizer text
model_inputs = tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors = 'pt')

# Generate
generated_tokens = model.generate(
    model_inputs["input_ids"].to(model.device),
    attention_mask=model_inputs["attention_mask"].to(model.device),
    decoder_start_token_id = tokenizer.convert_tokens_to_ids("tp_XX"),
    **gen_kwargs,
)

# Extract text
decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

data = []
# Extract triplets
for idx, sentence in enumerate(decoded_preds):
    print(f'Prediction triplets sentence {idx}')
    # print(sentence)
    tmp = extract_triplets_typed(sentence) 
    print(data)
    data.append(tmp)
    
file_path = f"{__file__}.json"
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"saving to {file_path}")
