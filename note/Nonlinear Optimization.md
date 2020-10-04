[toc]

# Element of Convex Analysis

## 1. Convex Set

- DEFINITION 2.1. A set $X\subset\R^n$ is called convex if for all $x^1\in X$ and $x^2\in X$ it contains all points

  $$\alpha x^1+(1-\alpha)x^2, 0<\alpha<1$$

  Convexity is preserved by all the operation of intersection.

- LEMMA 2.2. Let $I$ be an arbitrary index set. If the sets $X_i\in \R^n, i\in I$ ,are convex, then the set $X=\cap_{i\in I}X_i$ is convex.

- LEMMA 2.3. Let $X$and $Y$ be convex sets in $\R^n$ and let $c$ and $d$ be real numbers. Then the set $Z=cX+dY$ is convex.(Minkowski sum)

- DEFINITION 2.4. A point $x$ is called a convex combination of points $x^1,...,x^m$ if there exist $a_1\ge0,...a_m\ge0$ such that

  $$x=a_1x^1+a_2x^2+...+a_mx^m$$

  and

  $$a_1+a_2+...+a_m=1$$

- DEFINITION 2.5. The convex hull of the set $X$ (denoted by $convX$) is the intersection of all convex sets containing $X$.

- LEMMA 2.6. The set $conv X$ is the set of all convex combinations of points of $X$ .

- LEMMA 2.7. If $X\subset\R^n$ , then every element of $convX$ is a convex combination of at most $n+1$ points of $X$ .

- LEMMA 2.8. If $X$ is convex, then its interior int $X$ and its closure $\bar X$ are convex.

