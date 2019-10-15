from time import sleep
from json import dumps
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda x: dumps(x).encode('UTF-8'))

for i in range(1000):
    data = {'number': i}
    producer.send('test', value=data)
    sleep(5)

{
'CARD_ACCEPTOR_ACTIVITY':'4900',
'CODE_ACTION':'000',
'POS_ENTRY_MODE':'000',
'PROCESSING_CODE':'004000',
'TARGET':'0',
'TRANSACTION_AMOUNT':10700,
'TERM_CARD_READ_CAP':'2',
'TERM_CH_VERI_CAP':'9',
'TERM_CARD_CAPTURE_CAP':'0',
'TERM_ATTEND_CAP':'1',
'CH_PRESENCE_IND':'1',
'CARD_PRESENCE_IND':'1',
'TXN_CARD_READ_IND':'1',
'TXN_CH_VERI_IND':'0',
'TXN_CARD_VERI_IND':'0',
'TRACK_REWRITE_CAP':'0',
'TERM_OUTPUT_IND':'4',
'PIN_ENTRY_IND':'9'
}