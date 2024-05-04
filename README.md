# wellbot-healthcare-chatbot
 WellBot: An NLP-powered Health-care and Mental Health Support Chatbot.

WellBot isn't just another chatbot; it's a compassionate listener and a knowledgeable advisor rolled into one. Through open and non-judgmental discussions, WellBot invites users to share their health concerns, understanding the nuances of each conversation through advanced natural language processing. Whether you're seeking advice for managing stress, coping with chronic pain, or navigating mental health challenges, WellBot provides tailored recommendations to support your well-being journey.


Please read through the Report.pdf file for more detailed description of the chatbot.

Installation 

Run the following command to install all the dependencies : 

pip install -r .\requirements.txt

Run the application by executin the following command : 

cd .\Project\

python.exe .\chat_bot.py

Sample prompts for interaction with chatbot :

-----------------------------------HealthCare ChatBot-----------------------------------
Please enter the following information : 

Name : ->Kevin   

Age : -> 25

Gender : -> Male

Medical History : -> None

Hello,  Kevin!

Enter the (3-4) main symptoms that you are experiencing for accurate diagnosis  ->cold, cough, headache, runny nose

Please select the symtpoms by entering their ids(0 ,1 ,2..etc):
0 ) cold
1 ) cough
2 ) headache
3 ) runny_nose
Confirm the ones you meant (0 - 3):  0 , 1, 2, 3

How many days have you been experiencing this symptom ? : 3

In addition to that, are you experiencing any of the following symptoms : 

continuous_sneezing ? : yes
chills ? : yes
fatigue ? : yes
fever ? : yes
swelled_lymph_nodes ? : no
malaise ? : yes
phlegm ? : yes
throat_irritation ? : yes
redness_of_eyes ? : no
sinus_pressure ? : no
congestion ? : yes
chest_pain ? : no
loss_of_smell ? : no
muscle_pain ? : no

It might not be that bad but you should take precautions.
You may have Common Cold
The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold.
You don't have anything related to your past medical history.

Take following measures :
1 ) drink vitamin c rich drinks
2 ) take vapour
3 ) avoid cold food
4 ) keep fever in check
5 ) For more information visit - https://en.wikipedia.org/wiki/Common_cold

For the specialty Family Physician, the following doctors are recommended:
Dr.Matthew Pabis,MD - Specialty: Family Physician, More Info: https://www.zocdoc.com/doctor/matthew-pabis-md-221792
Dr.Rajiv Agashiwala,DO - Specialty: Family Physician, More Info: https://profiles.mountsinai.org/rajiv-agashiwala
Dr.Diana Nassar,MD - Specialty: Family Physician, More Info: https://www.reside.health/providers/diana-nassar-md

Loading your wellness assisstant ....

Have any more questions? Would you like to chat with our Wellbot assistance?
Yes

Hello! I'm WellBot, your Mental Health Assistant. How are you feeling today?

User: I feel alright!
Chatbot: It's great to hear that you're feeling good! Keeping up with positive vibes is wonderful.

Chatbot: Do you have anymore questions?
User: Yes

Chatbot: Ask a question or enter 'quit' to exit.

User: How do I improve my mental health?
Chatbot: The most important thing is to talk to someone you trust. This might be a friend, colleague, family member, or GP. In addition to talking to someone, it may be useful to find out more information about what you are experiencing. These things may help to get some perspective on what you are experiencing, and be the start of getting help.

User: How do I improve my physical strength?
Chatbot: Taking care of your physical health is also good for your mental health. It's more important than ever to keep yourself healthy.
 Try to eat as well as you can. It may be easier to reach for unhealthier comfort foods and snacks while you spend more time at home, but try to keep a balanced approach. When you stock up on groceries, don’t ignore fresh fruit and vegetables—we still have everything we need to prepare food.
 
User: quit
Chatbot: Goodbye! Thanks for chatting!
