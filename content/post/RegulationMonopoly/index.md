
## Optimal regulation of a monopoly
---
**Principle**
In a market, we achieve **allocative efficiency** when all units of production that generate a positive surplus are produced.

- In other words: the consumer's willingness to pay for an additional unit should be at least as high as the marginal cost of production.
- Efficient allocation of resource = marginal cost pricing.

**A simple example**
Suppose that $C(q)=F+c q$. What is the efficient price? What is the firm's profit at this efficient price?
- Efficient price: $p^{\star}=c$
- Leads to a loss for the monopoly: $\pi^{\star}=-F$
- In the previous example, we obtained $\pi^{\star}=-F<0$ !
- There is a budget balance problem $\rightarrow$ **the optimal regulation is not feasible**
- A solution: give the firm a subsidy of $F$
- Problem?
	- Subsidies may be prohibited by law
	- To get $F$, the regulator or the government should raise a tax, which will lead to a loss of efficiency... higher or lower than the efficiency loss that the regulator is supposed to eliminate
	- A budget transfer from the State to the regulated firm introduces a risk of "rent seeking": we talk about "regulator capture"

### Regulation with a budget balance constraint
---

Maximize social welfare under the constraint that the regulated firm has a balanced budget ($\pi \geq 0$)

**Single-product monopoly case?**
- Simple: average cost pricing

**Definition**
Average cost pricing in a single product monopoly is a regulatory requirement that the monopolist charge a price equal to the average total cost of production. This means that the monopolist must earn a normal profit, which is just enough to cover all of its costs and stay in business.
To calculate the average total cost of production, the monopolist must add up all of its costs, including fixed and variable costs, and divide by the total number of units produced. The average total cost will vary depending on the monopolist's cost structure, but it will generally be higher than the marginal cost of production.

**Multi-product monopoly case?**
- More complex: there are many combinations of prices and quantities such that the monopoly firm makes a non-zero profit
- Optimal price combination: "Ramsey-Boiteux" pricing
- Ramsey-Boiteux prices are proportional (but lower) to the inverse elasticity: the idea is to cover the fixed costs by **charging more for the least elastic services.**


### Alternatives to regulation
---
Costs of regulation:
- Information asymmetries (costs, demand)
- Direct costs of regulation (staff of regulatory agency, etc.)
- Risks of capture
Other solutions than regulation?
- Competition "à la Demsetz"
- Contestable markets
- Intermodal competition

#### Competition "à la Demsetz"
---
- If competition in the market is not possible, we can organize an auction to grant the market to the firm offering the "highest bid" (i.e., proposing the lowest price for the good)
- Auction for the market = competition "for the market" instead of competition "in the market"
- In a single product industry, if there is no collusion between the bidders, and if production inputs are available to all at a competitive price $\rightarrow$ competition "à la Demsetz" should lead to average cost pricing

#### Contestable markets
---
- _Theory of Baumol, Panzar and Willig (1982)
- Competition for the market should lead to the optimum with budget balance without public intervention (such as bidding for the market), if there are no sunk costs
- **Sunk costs** = fixed costs that cannot be recouped when production stops
- If the monopoly sets a price higher than marginal cost, competitors will enter and take over the market by setting a slightly lower price ("hit and run" strategy)

#### Intermodal Competition
---
Competition between different "modes" of production
Examples:
- Competition between different modes of transportation: rail versus road for freight
- Competition between different electronic communication networks: telecom networks versus cable TV or satellite networks

# Price discrimination
---
**Definition of price discrimination**
The practice of charging different prices for the same good (or similar goods), the selling price depending on: the quantity purchased, the characteristics of the buyer, or other contract terms
**Examples:**
- Student price
- Airline fares ("yield management")
- Volume discounts ("2nd product offered")
- Vouchers ...

**Question:** How do we know if there is price discrimination?

## Test
---
There is price discrimination if difference in price between two versions of a good cannot be explained by a difference in cost
**Stigler test (1987):**
$$
\frac{p_1}{p_2} \neq \frac{c_1}{c_2}
$$
**Philips test (1983):**
$$
\left(p_1-c_1\right) \neq\left(p_2-c_2\right)
$$

## Conditions for price discrimination
---
Conditions for price discrimination:
1. Firms should have market power
2. Consumers should have different willingness to pay and firms should be able to identify them directly or indirectly (self-selection)
3. Resale opportunities should be limited

Resale (or arbitrage) is difficult:
- If the good is a service
- If the warranty applies only to the buyer
- If transaction costs are high (storage costs, search costs...)
- If there is a legal restriction on resale

## Pigou classification
---
Pigou (1920) identifies three forms of price discrimination:
- First degree discrimination (or personalized pricing)
- Third degree discrimination (or group pricing)
- Second degree discrimination (or versioning, or menu pricing). Includes volume discounts (and all forms of non-linear pricing)

These three forms of price discrimination require some level of information about consumers, in decreasing order (1st degree $3rd$ degree $2nd$ degree)

### First-degree price discrimination
---
**Definition (Tirole, 1988)**
The producer captures the entire consumer surplus


Examples of first-degree price discrimination? $\rightarrow$ Bazaar, fortune teller, Amazon experience (2000)
- In a bazaar, for example, sellers may negotiate with each customer individually to find the highest price that they are willing to pay for a particular item. This is a common practice in many developing countries, where markets are often less competitive and sellers have more bargaining power.
- Fortune tellers also often use first-degree price discrimination. They may charge customers different prices based on their perceived income, wealth, or willingness to pay. For example, a fortune teller may charge a wealthy customer more for a reading than they would charge a customer who appears to be less wealthy.
- Amazon Experience (2000) was a short-lived program that allowed Amazon customers to bid on prices for items. This is a textbook example of first-degree price discrimination, as Amazon was charging each customer the maximum price that they were willing to pay.
What is the deadweight loss? $\rightarrow$ No deadweight loss...
**Remark**: If a monopoly implements first-degree price discrimination, allocative efficiency is reached

An example of first-degree price discrimination
First-degree price discrimination is possible when consumers consume more than one unit of the good or service
Let's consider a monopoly telecommunication operator
- All consumers are identical
- The utility of making $q$ phone calls is $u(q)$
- The monopoly sets a two-part tariff $T(q)=f+p q$
- $f=$ subscription, $p=$ price per call (or minute)

What is the optimal price for the monopolist? How can it implement first-degree price discrimination?

1. First step: once a consumer has subscribed to the service, he chooses the number of calls $q$ he wants to make to maximize his net utility, $u(q)-p q$, and obtains the following utility from making this optimal number of calls:
$$
v(p)=\max _q\{u(q)-p q\}
$$
2. Second step: the monopolist anticipates the consumer's optimal number of calls. It sets the subscription price so that the utility of making calls is higher than the subscription price: $v(p) \geq f$
3. Third step: let's write $q(p)$ the demand for calls. The monopoly problem is:
$$
\underset{p, f}{\max \pi}=(p-c) q(p)+f
$$
under the constraint that
$$
f \leq v(p)
$$
Let's replace $f$ by $v(p)$ and differentiate wrt $p(\mathrm{CPO})$ :
$$
q(p)+(p-c) \frac{\partial q(p)}{\partial p}+\underbrace{\frac{\partial v(p)}{\partial p}}_{=-q(p)}=0
$$
we have therefore
$$
(p-c) \frac{\partial q(p)}{\partial p}=0
$$
such that
$$
p^*=c
$$
**Result**
 $$
\text { The optimal price is such that } p^*=c \text { and } f^*=v\left(p^*\right)
$$

Intuition:
- The monopoly sets a price for calls that maximizes consumer surplus
- And extracts all the surplus with the subscription price
Remark: all consumers pay the same price

### Third-degree price discrimination
---
**Definition**
Third-degree price discrimination occurs when the monopoly sets a different price for each of its customer segments and is able to identify which segment each of its customers belongs to.

Example
For example, suppose a monopoly operates in different geographical markets The monopoly sets its price in each market so that the marginal revenue is the same in all markets and equal to marginal cost:$$M R_1=M R_2=\cdots=m c$$
This can be written using the Lerner index:
$$\frac{p_i-m c}{p_i}=\frac{1}{\epsilon_i}$$
The price of the good is lower in the market where the demand is the most elastic

### Second-degree price discrimination
---
**Definition:** Second-degree price discrimination occurs when the monopoly sets a different price for each of its customer segments and is unable to identify which segment each of its customers belongs to

We also talk about discrimination by self-selection, versioning, or menu pricing.
Idea:
- The monopoly cannot identify the customers
- But it knows the distribution of customer types in the population
- The monopoly can define an offer to discriminate between the different types of customers
- How? What constraints should be taken into consideration?






****