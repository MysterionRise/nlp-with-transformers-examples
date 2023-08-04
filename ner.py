import spacy
from spacy import displacy

NER = spacy.load("en_core_web_trf")
# please run python -m spacy download en_core_web_lg

raw_text = (
    "West Germany (German: Westdeutschland) is the colloquial English "
    "term used to indicate the Federal Republic of Germany (FRG; German: Bundesrepublik Deutschland "
    "[ˈbʊndəsʁepuˌbliːk ˈdɔʏtʃlant] (listen), BRD) between its formation on 23 May 1949 "
    "and the German reunification through the accession of East Germany on 3 October 1990. "
    "During the Cold War, the western portion of Germany and the associated territory of West Berlin "
    "were parts of the Western Bloc. West Germany was formed as a political entity during the Allied "
    "occupation of Germany after World War II, established from 12 states formed in the three Allied zones "
    "of occupation held by the United States, the United Kingdom, and France. "
    "The FRG's provisional capital was the city of Bonn, and the Cold War era country is retrospectively "
    "designated as the Bonn Republic (Bonner Republik).[4]"
)
ner_text = NER(raw_text)

for word in ner_text.ents:
    print(word.text, word.label_)

print(displacy.render(ner_text, style="ent"))
