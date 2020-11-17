import numpy as np 

def test(x):
    probability=0.9 #accuracy of the test. 
    #90% probability of a correct classification:
    return x if np.random.rand() < probability else 1-x         

population=np.zeros(100)
population[0]=1 # 1% is sick 

n=1000000 # we run 1,000,000 experiments
#pick a random sample of size 'n' where 1% are ones, the rest zero 
actual = np.random.choice(population,replace=True,size=n)
#apply the test to each entry
#there is a 90% probability of not flippling values  0<-->1  (correct classification)
test_result=np.array(list(map(test,actual)),dtype=int)

#P(negative test) = 1-P(positive test) 
#the sum is the total number of ones in test_result
p_negative=1-np.sum(test_result)/n

#P(probability of negative test | has covid) is the actuals that were flipped (miss classified) divided by those with negative test
p_negative_has_covid=np.sum(actual[test_result==0])/(np.sum(test_result==0))

print("(a) P(probability of negative test) = %.1f%% (b) P(probability of negative test | has covid) = %.2f"%(100*p_negative,100*p_negative_has_covid))
