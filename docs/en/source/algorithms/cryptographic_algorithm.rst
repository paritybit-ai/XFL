===============================
Cryptographic Algorithm
===============================

XFL uses PHE(partially homomorphic encryption), FHE(fully homomorphic encryption), 
one-time pad and other cryptographic techniques to encrypt sensitive data such as gradients, 
model parameters and intermediate parameters to ensure the security in a federation.


Homomorphic Encryption
===============================

Paillier
------------

Paillier [Paillier]_  cryptosystem, based on the Decisional Composite Residuosity Assumption, 
is a widly used PHE cryptosystem.

XFL uses the self-developed Paillier algorithm.
It is recommended to use the Paillier cryptosystem with a key length of not less than 2048 bits, 
and the computational security strength is not less than 112 bits.


CKKS
------------

CKKS [CKKS]_ is an efficient fully homomorphic encryption cryptosystem based on the hardness assumption of RLWE(Ring Learning with Errors). 
XFL calls the CKKS algorithm provided by TenSeal [TenSeal]_, which is based on Seal [Seal]_.
CKKS supports ciphertext addition and ciphertext multiplication, especially it supports SIMD(Single Instruction Multiple Data), making it highly efficient when performing batch operations.
CKKS allows a variety of parameter combinations, all of which meat high security.
According to the actual data, users can choose an appropriate set of parameters to achieve the optimal efficiency and accuracy requirements.
For example, in the demo of vertical logistic regression, the CKKS parameters are:

::

    "poly_modulus_degree": 8192, 
    "coeff_mod_bit_sizes": [60, 40, 40, 60],
    "global_scale_bit_size": 40

The above parameters can deal with input plaintext with batch size not greater than 4096, and the efficiency is highest when
the batch size is equal to 4096. The computational security strength is not less than 128 bits.


Cryptographic algorithms in secure aggregation
=================================================================

The secure aggregation algorithm [FedAvg]_ for horizontal fedration is an efficient aggregation method 
that involves Diffile-Hellman key exchange, secure pseudo-random number generator, and one-time-pad encryption method.


Diffile-Hellman key exchange
--------------------------------

The principle of key exchange is as follows:


Alice and Bob agree on a large prime number :math:`p` and a primitive root :math:`g`. Act the following:

.. image:: ../images/Diffie-Hellman_en.png

After that, both parties get the same random number. XFL supports parameters recommended in RFC 7919 [RFC7919]_ .
It is recommended to use a prime number with a bit length greater than or equal to 3072 bits :math:`p`, 
and the computational security strength is greater than or equal to 125 bits.


CSPRNG
-----------------------------------------------------------------

CSPRNG(Cryptographically secure pseudo-random number generator), a secure pseudo-random number generator generates random bit strings by inputting a random seed key.
XFL currently supports the hmac_drbg method specified in [SP800-90a]_ to generate secure pseudo-random numbers 
whose computational security strength is half of the output bits of the selected Hash algorithm.
It is recommended to use a Hash algorithm with an output bit length greater than or equal to 256 bits, 
such as sha256, and the input random seed bit length of the secure pseudo-random number generator should be no less than the output bit length of the Hash algorithm.


One time pad
--------------------------------

One-time pad is a classic encryption algorithm, which is characterized by using a different key for each encryption. 
There are various forms of one-time pad, such as bitwise exclusive, modulo addition, etc. 
XFL adopts the modulo addition form. Currently, the modulus supports :math:`2^{64}` and :math:`2^{128}`. 
The statistical security parameters are 64-bit and 128-bit respectively.


:References:

.. [Paillier] Paillier P. Public-key cryptosystems based on composite degree residuosity classes[C]//International conference on the theory and applications of cryptographic techniques. Springer, Berlin, Heidelberg, 1999: 223-238.
.. [CKKS] Cheon J H, Kim A, Kim M, et al. Homomorphic encryption for arithmetic of approximate numbers[C]//International conference on the theory and application of cryptology and information security. Springer, Cham, 2017: 409-437.
.. [TenSeal] https://github.com/OpenMined/TenSEAL.
.. [Seal] https://github.com/microsoft/SEAL.
.. [RFC7919] Gillmor D. Negotiated finite field Diffie-Hellman ephemeral parameters for transport layer security (TLS)[R]. 2016..
.. [FedAvg] Bonawitz K, Ivanov V, Kreuter B, et al. Practical secure aggregation for privacy-preserving machine learning[C]//proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security. 2017: 1175-1191.
.. [SP800-90a] Barker E, Kelsey J. NIST special publication 800-90a recommendation for random number generation using deterministic random bit generators[J]. 2012.