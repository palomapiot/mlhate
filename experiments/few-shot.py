import re

import json_repair, random
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm

print('Load data', flush=True)
df = pd.read_csv('data/MetaHate<lang>_test.tsv', sep='\t')

instruction = "Determine whether a social media message is hateful or not. Respond with either a True for hate speech or False for neutral. Do not include nothing else than True or False."
sentences = df['text'].to_list()

messages = []
"""few_shot = [
    {
        "from": "human",
        "value": instruction + " <Message>Calla ya subnormal de puta mierda @Lola_Cubero</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Es cierto que empalar a alguien y hacer bullying no son lo mismo, pero una cosa sí puede llevar a la otra. El bullying se puede medir con violentómetro. Lo que comienza con burlas, risas, y “chistes pesados” (que de chistes no tienen nada), sí puede terminar en la muerte.</Message>"

    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>El cine americano es genial y el español es un sector de rojos de mierda chupasubvenciones. Muerte a la inteligencia.</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Hay gente muy subnormal xD</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>@CreoqueAlex pues hazme los deberes furcia barata</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Escuchar a un hombre hablar de la igualdad de género y que de repente use la palabra “feminazi”. Bai amigo, perdiste toda credibilidad y atención.</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Oportunidades? No vamos a cargar los españoles con las oportunidades que deberían tener los inmigrantes en sus países de origen.</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Yo desde el día que saliste del armario pero eso fue en junio pero ese video me suena ;) y creo que igual de antes tengo 15 años y eres mi inspiración ✨ gracias por todo y un besico para la lucre para ti y alba besos desde pamplona 🍂🍂</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
]"""
few_shot = [
    {
        "from": "human",
        "value": instruction + " <Message>Cala xa subnormal de merda @lola_cubero</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>É certo que empar a alguén e facer acoso escolar non é o mesmo, pero unha cousa pode levar á outra. O acoso escolar pódese medir co violentómetro. O que comeza con burlas, risas e 'bromas pesadas' (que non teñen nada de bromas), poden acabar coa morte.</Message>"

    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>O cine americano é xenial e o español é un sector de merda vermella. Morte á intelixencia.</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Hai xente moi subnormal xD</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>@CreoQueAlex pois fai os meus deberes furcia barata</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Escoita a un home falar sobre a igualdade de xénero e de súpeto usa a palabra 'feminazi'. Bai amigo, perdeu toda a credibilidade e a atención.</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Oportunidades? Non imos cargar os españois coas oportunidades que deberían ter os inmigrantes nos seus países de orixe.</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Desde o día en que deixaches o armario, pero iso foi en xuño, pero ese vídeo paréceme;) e creo que teño 15 anos e ti es a miña inspiración ✨ Grazas por todo e un bico polo lucre para ti e os bicos de Dawn de Pamplona 🍂🍂</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
]
"""few_shot = [
    {
        "from": "human",
        "value": instruction + " <Message>@Bidenshairplugs He looks like a fag and his time line is boring Zzzzzz</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Right time to by Financial companies? Bharat Finance, LT Finance...</Message>"

    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Hitler didn't finish it. Can u. If a nigger ur Jew confronts u in the street what then.</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Where do u see the crude heading towards march end 2017 ?</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>@CecileRichards Women aren't leading anything. Some radical brainwashed simpleton idiots are screaming. Who cares what they think?</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>You deserve this and rest what people write or will write about you: http://antandrus.tripod.com and encyclopediadramatica rules!</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Stupid whore, I'd rape that bloody cunt and pump a load deep inside you.</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>The fact that Jenny is raped while unconscious during the course of the film shows that Jenny may not be aware of all of the sexual encounters she's had.  She may have been raped before, and so her claim that she had only ever had sex with Telly may be false (without her even being aware that it's false.)  I always thought that was the whole point of the rape scene — to show that naive beliefs held by these kids can't be trusted.  Telly's belief that he's only ever slept with virgins is probably false (unless he slept with a virgin who had contracted HIV some other way) and so is Jenny's belief that Telly was her only partner, or that in a culture of random drugged-out sex and casual rape anybody can even be sure who they've slept with.  I've modified the article accordingly.</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
]"""
"""few_shot = [
    {
        "from": "human",
        "value": instruction + " <Message>Cala a boca, sua puta estúpida @Lola_Cubero</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>É verdade que empalar alguém e praticar bullying não são a mesma coisa, mas uma coisa pode levar à outra. O bullying pode ser medido com um medidor de violência. O que começa com provocações, risos e “piadas de mau gosto” (que não são de todo piadas), pode acabar em morte.</Message>"

    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>O cinema americano é ótimo e o cinema espanhol é um sector de merda, de saloios sugadores de subsídios. Morte à inteligência.</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Algumas pessoas são muito subnormais xD</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>@CreoqueAlex faz os meus trabalhos de casa puta barata</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Ouvir um homem falar de igualdade de género e, de repente, usar a palavra “feminazi”. Bai dude, perdeste toda a credibilidade e atenção.</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Oportunidades? Nós, espanhóis, não vamos ser sobrecarregados com as oportunidades que os imigrantes deveriam ter nos seus países de origem.</Message>"
    },
    {
        "from": "gpt",
        "value": "True"
    },
    {
        "from": "human",
        "value": instruction + " <Message>Eu tenho desde o dia em que saíste do guarda roupa mas isso foi em junho mas esse vídeo lembra-me alguma coisa ;) e acho que tenho 15 anos e tu és a minha inspiração ✨ obrigada por tudo e um beijo de lucre para ti e alba beijos de pamplona 🍂🍂</Message>"
    },
    {
        "from": "gpt",
        "value": "False"
    },
]"""


for sentence in sentences:
    message = few_shot + [{"from": "human", "value": instruction + " <Message>" + sentence + "</Message>"}]
    messages.append(message)


print('Load saved model', flush=True)
max_seq_length = 4096
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="hf_token"
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="<llama-3, mistral>",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

FastLanguageModel.for_inference(model)

generated = []
for message in tqdm(messages):
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.01,
        top_p=0.1,
        top_k=5
    )
    result = tokenizer.batch_decode(outputs)
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    #pattern = r"\[\/INST\](.*?)<\/s>"
    matches = re.findall(pattern, result[0], re.DOTALL)
    output = matches[-1].replace('\r', '').replace('\n', '')
    generated.append(output)

df['prediction'] = generated
df.to_csv('data/<model>-few-shot-<lang>.tsv', sep='\t', index=False)
print('saved!', flush=True)
