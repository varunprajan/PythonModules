from scipy.special import binom
import itertools as it

combsize = 5
decksize = 52
suitnum = 4
suitsize = decksize//suitnum

# N is the total number of cards in the deck
# K is the total number of possible (desirable) cards (e.g. 13 for a flush)
# k is the number of desirable cards drawn
# n is the total number of cards drawn

# I don't think excluding probability of better combinations from lesser combinations makes sense
# (E.g., straight flush from flush; two pair from pair)
# (Leads to weird results when n approaches 52)

# straight
def unnumbered_straight():
    return (suitsize-3)*numbered_straight(n=combsize)

def numbered_straight(n):
    notprob = 0
    kcombinations = it.product(range(suitnum+1),repeat=combsize)
    goodkvec = tuple(it.repeat(1,combsize))
    Kvec = tuple(it.repeat(suitnum,combsize))
    for kvec in kcombinations:
        if 0 in kvec:
            notprob += multi_hyper_poker(kvec,Kvec,n)
    return 1 - notprob

# pairs, two pairs, three of a kind, four of a kind, full house
def unnumbered_full_house():
    return suitsize*(suitsize-1)*numbered_full_house(n=combsize)

def numbered_full_house(n):
    return numbered_two_pair_or_more(3,2,n)

def unnumbered_four_kind():
    return suitsize*numbered_four_kind(n=combsize)

def numbered_four_kind(n):
    return hyper_poker(k=4,K=suitnum,n=n)
    
def unnumbered_three_kind():
    return suitsize*numbered_three_kind(n=combsize)

def numbered_three_kind(n):
    return numbered_pair_or_more(3,n)

def unnumbered_two_pair():
    return binom(suitsize,2)*numbered_two_pair(n=combsize)
    
def numbered_two_pair(n):
    return numbered_two_pair_or_more(2,2,n)

def unnumbered_pair():
    return suitsize*numbered_pair(n=combsize) - unnumbered_two_pair()
    
def numbered_pair(n):
    return numbered_pair_or_more(2,n)

def numbered_pair_or_more(kmin1,n):
    kexact = (hyper_poker(k=i,K=suitnum,n=n) for i in range(kmin1,suitnum+1))
    return sum(kexact)    
    
def numbered_two_pair_or_more(kmin1,kmin2,n):
    k1 = range(kmin1,suitnum+1)
    k2 = range(kmin2,suitnum+1)
    kcombinations = it.product(k1,k2)
    Kvec = (suitnum,suitnum)
    prob = 0
    for kvec in kcombinations:
        prob += multi_hyper_poker(kvec,Kvec,n)
    return prob    

# flushes
def unsuited_royal_flush():
    return suitnum*suited_royal_flush(n=combsize)

def suited_royal_flush(n):
    return numbered_suited_straight_flush(n)

def unsuited_straight_flush():
    return suitnum*unnumbered_suited_straight_flush()

def unnumbered_suited_straight_flush():
    return (suitsize-3)*numbered_suited_straight_flush(n=combsize)
    
def numbered_suited_straight_flush(n):
    return hyper_poker(k=combsize,K=combsize,n=n)

def unsuited_flush():
    return suitnum*suited_flush(n=combsize)

def suited_flush(n):
    kexact = (hyper_poker(k=i,K=suitsize,n=n) for i in range(combsize,suitsize+1))
    return sum(kexact)

# basics
def hyper_poker(k,K,n):
    return hypergeometric(k=k,K=K,n=n,N=decksize)
    
def hypergeometric(k,K,n,N):
    return binom(K,k)*binom(N-K,n-k)/binom(N,n)

def multi_hyper_poker(kvec,Kvec,n):
    return multi_hypergeometric(kvec=kvec,Kvec=Kvec,n=n,N=decksize)
    
def multi_hypergeometric(kvec,Kvec,n,N):
    kremainder = n - sum(kvec)
    Kremainder = N - sum(Kvec)
    numer = (binom(K,k) for (k,K) in zip(kvec,Kvec))
    return product(numer)*binom(Kremainder,kremainder)/binom(N,n)
    
def product(iterable):
    product = 1
    for x in iterable:
        product *= x
    return product