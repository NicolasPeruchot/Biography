from transformers import T5TokenizerFast, TFT5ForConditionalGeneration

model = TFT5ForConditionalGeneration.from_pretrained("./model")
tokenizer = T5TokenizerFast.from_pretrained("t5-small")

A=['Nicolas Peruchot', 'data scientist','french','Berlin','OneFootball','is']
val='|'.join(A)
tok=tokenizer(val, return_tensors="tf").input_ids
print(val)
outputs = model.generate(tok)
outputs1=tokenizer.decode(outputs[0]).replace('<pad> ','').replace('</s>',"")
print(outputs1)
