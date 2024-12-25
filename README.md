# BART_128
This is a BART task with LSL code. The builder version also exists. This experiment was created via PsychoPy ver. 2024.2.4. 
LSL code uses 3 types of labels for marking

The BART works by creating an array of numbers between 1 and 128. At each pump, a random number from this array is selected and deleted. If the number chosen is equal to 1, the balloon pops. If not, participant can continue to pump the balloon and lose their earnings for that balloon. The probability of popping the balloon increases by 1 with each pump. Participant can choose to bank in their earnins any time. 

1 = balloon start

2 = balloon pop

3 = balloon banked

Automatic marking is still new and is in development. The code works via lab recorder, but I need to test its accuracy. 
