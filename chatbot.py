from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#object for storing the conversation history
conversation_history = []

while True:
    #turn the list into a string for processing by the model
   
    history_string = "\n".join(conversation_history)
    
    input_text ="hello, how are you doing?"
    #encode your inputs as tokens so that they can be passed to the model.
    
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    
    #this contains pretrained vocabulary files
    tokenizer.pretrained_vocab_files_map
    #pass the inputs to the model and generate a response
    outputs = model.generate(**inputs)
    #Decode the ouputs tokens into plain text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)
    #Update the conversation history each time
    conversation_history.append(input_text)
    conversation_history.append(response)