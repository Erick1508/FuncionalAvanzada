\documentclass[11pt,fleqn]{article}

\usepackage{tikz}
\usepackage{multicol}
\usepackage{latexsym}
\usepackage{array}
\usepackage[english,spanish]{babel}
\usepackage{lmodern}
\usepackage{listings}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[colorlinks=true,urlcolor=blue]{hyperref}
\usepackage{xcolor}

\usepackage{algorithmic}
\usepackage{algorithm}

\usetikzlibrary{positioning,shapes,folding,positioning,shapes,trees}

\hypersetup{
  colorlinks=true,
  linkcolor=blue,
  urlcolor=blue
}

\definecolor{brown}{rgb}{0.7,0.2,0}
\definecolor{darkgreen}{rgb}{0,0.6,0.1}
\definecolor{darkgrey}{rgb}{0.4,0.4,0.4}
\definecolor{lightgrey}{rgb}{0.95,0.95,0.95}



\lstset{
   language=Haskell,
   gobble=2,
   frame=single,
   framerule=1pt,
   showstringspaces=false,
   basicstyle=\footnotesize\ttfamily,
   keywordstyle=\textbf,
   backgroundcolor=\color{lightgrey}
}

\long\def\ignore#1{}

\begin{document}

\title{CI4251 - Programación Funcional Avanzada \\ Tarea 1}

\author{Erick Marrero\\
09-10981\\
\href{mailto:emhn@usb.ve}{<09-10981@usb.ve>}}

\date{Mayo 01, 2015}

\maketitle

\pagebreak

\section{Machine Learning}

\subsection{Definiciones Generales}

Para el desarrollo de la solución, serán necesarios los módulos


\begin{lstlisting}
> import Data.List
> import Data.Functor
> import Data.Monoid
> import Data.Foldable (foldMap)
> import Data.Tree
> import Data.Maybe (fromJust)
\end{lstlisting}

Las técnicas de \emph{Machine Learning} operan sobre conjuntos
de datos o muestras. En este caso, existe un conjunto de muestras
que serán usadas para ``aprender'', y así poder hacer proyecciones
sobre muestras fuera de ese conjunto. Para user el método de
regresión lineal multivariable, las muestras son \emph{vectores}
$(x_1,x_2,\ldots,x_n)$ acompañados del valor asociado $y$
correspondiente.

En este ejercicio, nos limitaremos a usar vectores de dos variables,
pero usaremos un tipo polimórfico basado en listas, para poder
utilizar \texttt{Float} o \texttt{Double} según nos convenga, y
modelar vectores de longitud arbitraria. Su programa no puede
hacer ninguna suposición sobre la longitud de los vectores, más
allá de que todos son del mismo tamaño.

Así, definiremos el tipo polimórfico

\begin{lstlisting}
> data Sample a = Sample { x :: [a], y :: a }
>      deriving (Show)
\end{lstlisting}

\newpage

Teniendo una colección de muestras como la anterior, el algoritmo
calcula una \emph{hipótesis}, que no es más que un \emph{vector}
de coeficientes $(\theta_0, \theta_1, \ldots, \theta_n)$ tal que
minimiza el error de predicción
$(\theta_0 + \theta_1 \times x_1 + \ldots + \theta_n x_n - y)$
para toda la colección de muestras.


\begin{lstlisting}
> data Hypothesis a = Hypothesis { c :: [a] }
>      deriving (Show)
\end{lstlisting}

En el caso general, asegurar la convergencia del algoritmo en un
tiempo razonable es hasta cierto punto ``artístico''. Sin entrar en
detalles, es necesario un coeficiente $\alpha$ que regule
cuán rápido se desciende por el gradiente

\begin{lstlisting}
> alpha :: Double
> alpha = 0.03
\end{lstlisting}

También hace falta determinar si el algoritmo dejó de progresar,
para lo cual definiremos un márgen de convergencia $\epsilon$
muy pequeño

\begin{lstlisting}
> epsilon :: Double
> epsilon = 0.0000001
\end{lstlisting}

Finalmente, el algoritmo necesita una hipótesis inicial, a partir
de la cual comenzar a calcular gradientes y descender hasta encontrar
el mínimo, con la esperanza que sea un mínimo global. Para nuestro
ejercicio, utilizaremos

\begin{lstlisting}
> guess :: Hypothesis Double
> guess = Hypothesis { c = [0.0, 0.0, 0.0] }
\end{lstlisting}

\subsection{Muestras de Entrenamiento}

En este archivo se incluye la definición

\begin{lstlisting}
> training :: [Sample Double]
\end{lstlisting}

\ignore{
\begin{code}
training = [
  Sample { x = [  0.1300098690745405, -0.2236751871685913 ], y = 399900 },
  Sample { x = [ -0.5041898382231769, -0.2236751871685913 ], y = 329900 },
  Sample { x = [  0.502476363836692, -0.2236751871685913 ], y = 369000 },
  Sample { x = [ -0.7357230646969468, -1.537766911784067 ], y = 232000 },
  Sample { x = [  1.257476015381594, 1.090416537446884 ], y = 539900 },
  Sample { x = [ -0.01973172848186497, 1.090416537446884 ], y = 299900 },
  Sample { x = [ -0.5872397998931161, -0.2236751871685913 ], y = 314900 },
  Sample { x = [ -0.7218814044186236, -0.2236751871685913 ], y = 198999 },
  Sample { x = [ -0.7810230437896409, -0.2236751871685913 ], y = 212000 },
  Sample { x = [ -0.6375731099961096, -0.2236751871685913 ], y = 242500 },
  Sample { x = [ -0.07635670234773261, 1.090416537446884 ], y = 239999 },
  Sample { x = [ -0.0008567371932424295, -0.2236751871685913 ], y = 347000 },
  Sample { x = [ -0.1392733399764744, -0.2236751871685913 ], y = 329999 },
  Sample { x = [  3.117291823687202,   2.40450826206236 ], y = 699900 },
  Sample { x = [ -0.9219563120780225, -0.2236751871685913 ], y = 259900 },
  Sample { x = [  0.3766430885792084,  1.090416537446884 ], y = 449900 },
  Sample { x = [ -0.856523008944131,  -1.537766911784067 ], y = 299900 },
  Sample { x = [ -0.9622229601604173, -0.2236751871685913 ], y = 199900 },
  Sample { x = [  0.7654679091248329,  1.090416537446884 ], y = 499998 },
  Sample { x = [  1.296484330711414,   1.090416537446884 ], y = 599000 },
  Sample { x = [ -0.2940482685431793, -0.2236751871685913 ], y = 252900 },
  Sample { x = [ -0.1417900054816241, -1.537766911784067 ], y = 255000 },
  Sample { x = [ -0.4991565072128776, -0.2236751871685913 ], y = 242900 },
  Sample { x = [ -0.04867338179108621, 1.090416537446884 ], y = 259900 },
  Sample { x = [  2.377392165173198,  -0.2236751871685913 ], y = 573900 },
  Sample { x = [ -1.133356214510595,  -0.2236751871685913 ], y = 249900 },
  Sample { x = [ -0.6828730890888036, -0.2236751871685913 ], y = 464500 },
  Sample { x = [  0.6610262906611214, -0.2236751871685913 ], y = 469000 },
  Sample { x = [  0.2508098133217248, -0.2236751871685913 ], y = 475000 },
  Sample { x = [  0.8007012261969283, -0.2236751871685913 ], y = 299900 },
  Sample { x = [ -0.2034483103577911, -1.537766911784067 ], y = 349900 },
  Sample { x = [ -1.259189489768079,  -2.851858636399542 ], y = 169900 },
  Sample { x = [  0.04947657290975102, 1.090416537446884 ], y = 314900 },
  Sample { x = [  1.429867602484346,  -0.2236751871685913 ], y = 579900 },
  Sample { x = [ -0.2386816274298865,  1.090416537446884 ], y = 285900 },
  Sample { x = [ -0.7092980768928753, -0.2236751871685913 ], y = 249900 },
  Sample { x = [ -0.9584479619026928, -0.2236751871685913 ], y = 229900 },
  Sample { x = [  0.1652431861466359,  1.090416537446884 ], y = 345000 },
  Sample { x = [  2.78635030976002,    1.090416537446884 ], y = 549000 },
  Sample { x = [  0.202993168723881,   1.090416537446884 ], y = 287000 },
  Sample { x = [ -0.4236565420583874, -1.537766911784067 ], y = 368500 },
  Sample { x = [  0.2986264579195686, -0.2236751871685913 ], y = 329900 },
  Sample { x = [  0.7126179335166897,  1.090416537446884 ], y = 314000 },
  Sample { x = [ -1.007522939253111,  -0.2236751871685913 ], y = 299000 },
  Sample { x = [ -1.445422737149154,  -1.537766911784067 ], y = 179900 },
  Sample { x = [ -0.1870899845743182,  1.090416537446884 ], y = 299900 },
  Sample { x = [ -1.003747940995387,  -0.2236751871685913 ], y = 239500 } ]
\end{code}
}

que cuenta con 47 muestras de entrenamiento listas para usar.


\subsection{ Funciones a desarrollar }

\subsubsection{Comparar en punto flotante}

En esta función se restan los valores para chequear si es 
despreciable o no

\begin{lstlisting}
> veryClose :: Double -> Double -> Bool
> veryClose v0 v1 = abs (v0 - v1) <= epsilon 
\end{lstlisting}


\subsubsection{Congruencia dimensional}

Para agregar un coeficiente, se hace un map para recorrer
cada muestra y se coloca un 1 adicional en x

\begin{lstlisting}
> addOnes :: [Sample Double] -> [Sample Double]
> addOnes = map agregar 
>  where agregar m = Sample {x = add (x m), y = y m}
>	add lis =  1:lis
\end{lstlisting}


\subsubsection{Evaluando Hipótesis}

Si tanto una hipótesis $\theta$ como una muestra $X$ son vectores
de $n+1$ dimensiones, entonces se puede evaluar la hipótesis en
$h_\theta(X) = \theta^TX$ calculando el producto punto
de ambos vectores.

Para ello se multiplica el vector 'h' con el vector 'x' componente a 
componente usando zipWith y luego se usa un foldl' para hacer las 
sumas de la lista resultante

\begin{lstlisting}
> theta :: Hypothesis Double -> Sample Double -> Double
> theta h s = foldl' (+) 0 ( zipWith (*) (c h) (x s))
\end{lstlisting}

Una vez que pueda evaluar hipótesis, es posible determinar cuán
buena es la hipótesis sobre el conjunto de entrenamiento. La calidad
de la hipótesis se mide según su \textbf{costo} $J(\theta)$ que no
es otra cosa sino determinar la suma de los cuadrados de los errores.
Para cada muestra $x^{(i)}$ en el conjunto de entrenamiento, se
evalúa la hipótesis en ese vector y se compara con el $y(i)$. La
fórmula concreta para $m$ muestras es
$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m}{(h_\theta(x^{(i)}) - y^{(i)})^2} $$

Para el cálculo del costo, se usa un foldr para recorrer las muestras y aplicar 
la fórmula correspondiente, y finalmente se divide entre la cantidad de
muestras

\begin{lstlisting}
> cost :: Hypothesis Double -> [Sample Double] -> Double
> cost h ss =
>	(foldr ((+).eval h) 0 ss) / (fromIntegral (length ss)*2)
>  	where eval hy sam = (theta hy sam - y sam)^2
\end{lstlisting}

\subsection{Bajando por el gradiente}

El algoritmo de descenso de gradiente por lotes es sorprendentemente
sencillo. Se trata de un algoritmo iterativo que parte de una
hipótesis $\theta$ que tiene un costo $c$, determina la dirección
en el espacio vectorial que maximiza el descenso, y produce una
nueva hipótesis $\theta'$ con un nuevo costo $c'$ tal que $c' \leq c$.
La ``velocidad'' con la cual se desciende por el gradiente viene
dada por el coeficiente ``de aprendizaje'' $\alpha$.

Dejando el álgebra vectorial de lado por un momento, porque no
importa para esta materia, es natural pensar que nuestro algoritmo
iterativo tendrá que detenerse cuando la diferencia entre $c$ y $c'$
sea $\epsilon-$despreciable.

Para calcular la nueva hipótesis, se hace un foldl' para ir recorriendo
las componentes de la hipótesis anterior, y el valor semilla es una
tupla que contiene una lista que guarda los valores calculados de la 
nueva hipótesis  y el otro valor de la tupla es la posición de la 
componente que se está modificando.
La función \texttt{calculo} es la que permite crear un nuevo valor a la 
lista y llama a \texttt{sumatoria} que hace un foldr para sumar todos los 
valores que retorna la función \texttt{funaux} que aplica la fórmula 
correspondiente. Y finalmente como es una tupla lo que se lleva, 
se toma el primero con ``fst'' y ``reverse'' para colocarlo en el orden
correcto.

\begin{lstlisting}
> descend :: Double -> Hypothesis Double -> [Sample Double]
>         -> Hypothesis Double
> descend alpha h ss = 
>  Hypothesis {c = reverse ( fst (
>    foldl' (calculo (length ss) alpha h ss ) ([],0) (c h)))}
>   	where calculo tam alpha hs ss (lis, num) h = 
>                (sumatoria tam alpha ss hs num : lis, num+1) 
>	sumatoria tam alpha ss hy num = c hy !! num - 
>		((foldr ((+).funaux hy num) 0 ss) 
>		* alpha/ fromIntegral tam) 
>	funaux hyp num s = ((theta hyp s) - y s)* (x s !! num)
\end{lstlisting}

Sea $\theta_j$ el $j-$ésimo componente del vector $\theta$
correspondiente a la hipótesis actual que pretendemos mejorar. La
función debe calcular, para todo $j$

$$\theta'_j \leftarrow \theta_j -
\frac{\alpha}{m}\sum_{i=1}^m{(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}}$$

donde $m$ es el número de muestras de entrenamiento.

La segunda parte de este algoritmo debe partir de una hipótesis
inicial y el conjunto de entrenamiento, para producir una lista
de elementos tales que permitan determinar, para cada iteración,
cuál es la hipótesis mejorada y el costo de la misma.

En la función \texttt{gd}, se uso ``unfoldr'' para ir construyendo la 
lista con nuevas hipótesis y nuevos costos en cada iteración, éstas
usan la función \texttt{calculo} para calcular los nuevos valores. Esto
se hace hasta que los costos seas despreciables usando la función
\texttt{veryClose} que es la condición de parada.


\begin{lstlisting}
> gd :: Double -> Hypothesis Double -> [Sample Double]
>    -> [(Integer,Hypothesis Double,Double)]
> gd alpha h ss = 
>    unfoldr (\(i,hypo,c) -> 
>    if veryClose c 
>	(cost (descend alpha hypo (addOnes ss)) (addOnes ss)) 
>         then Nothing
>	else 
>	  Just ((i,hypo,c),calculo i alpha hypo (addOnes ss))) 
>		(0,h,cost h (addOnes ss))
>  where calculo i alpha hy ss = 
>	(i+1,descend alpha hy ss, cost (descend alpha hy ss) ss)
\end{lstlisting}


\subsection{ Resultados Obtenidos}

Probar las funciones \texttt{veryClose}, \texttt{addOnes}
y \texttt{theta} es trivial por inspección. Para probar la
función \texttt{cost} tendrá que hacer algunos cálculos a
mano con muestras pequeñas y comprobar los resultados que
arroja la función. Preste atención que estas funciones
\emph{asumen} que las muestras ya incluyen el coeficiente
constante 1.

Probar la función \texttt{descend} es algo más complicado,
pero la sugerencia general es probar paso a paso si se
produce una nueva hipótesis cuyo costo es, en efecto, menor.


\begin{lstlisting}
>  *Main>  take 3 (gd alpha guess training)
> [(0,Hypothesis {c = [0.0,0.0,0.0]},6.559154810645744e10),
> (1,Hypothesis {c = [10212.379787234042,3138.9880129854737,
>			1623.7105153222735]},
>			6.1759853666230095e10),
> (2,Hypothesis {c = [20118.388180851063,6159.113611965673,
>			3148.136171427171]},
>			5.8164574323311745e10)]

\end{lstlisting}

y si se deja correr hasta terminar converge (el \emph{unfold}
\textbf{termina}) y los resultados numéricos en la última tripleta.
 
 
  
   
   
\begin{lstlisting}
>  *Main> last (gd alpha guess training)
>	(1213,Hypothesis {c = [340412.65957446716,
>		110631.04200819538,-6649.466000169089]},
>		2.0432800506028578e9)

\end{lstlisting}


Una vez que el algoritmo converge, obtenga la última hipótesis
y úsela para predecir el valor $y$ asociado al vector
$(-0.44127, -0.22368)$.

\begin{verbatim}
  ghci> let (_,h,_) = last (gd alpha guess training)
  ghci> let s = Sample ( x = [1.0, -0.44127,-0.22368], y = undefined }
  ghci> theta h s
  293081.8522224286
\end{verbatim}

\section{Monoids}

Construya una instancia \texttt{Monoid}
\emph{polimórfica} para \emph{cualquier} tipo comparable, tal que
al aplicarla sobre cualquier \texttt{Foldable} conteniendo 
elementos de un tipo concreto comparable, se retorne el máximo
valor almacenado, si existe. La aplicación se logra con la
función

\begin{verbatim}
foldMap :: (Foldable t, Monoid m) => (a -> m) -> t a -> m
\end{verbatim}

Para la construcción de este Monoid, se puede notar claramente que
tienen que ser elementos comparables entre ellos ya que se va a calcular
el máximo de los mismos. Como no se sabe con que estructura va a trabajar 
el Monoid, entonces puede ser que no se obtenga un resultado. Por ello
se trabajará con el Maybe. Por lo tanto, el neutro para la función \texttt{mempty} 
es el tipo Nothing y para la función \texttt{mappend} retorna el máximo
entre esos elementos colocando la marca Max.


\begin{lstlisting}
> newtype Max a = Max { getMax :: Maybe a }
>	deriving ( Eq ,Ord,Show)
>
> instance (Ord a) => Monoid ( Max a ) where
>	mempty                       = Max Nothing
>	mappend (Max x) (Max y) = Max $ max x y

\end{lstlisting}

Oriéntese con los siguientes ejemplos

\begin{verbatim}
ghci> foldMap (Max . Just) []
Max {getMax = Nothing}
ghci> foldMap (Max . Just) ["foo","bar","baz"]
Max {getMax = Just "foo"}
ghci> foldMap (Max . Just) (Node 6 [Node 42 [], Node 7 [] ])
Max {getMax = Just 42}
ghci> foldMap (Max . Just) (Node [] [])
\end{verbatim}

\section{Zippers}

Considere el tipo de datos

\begin{lstlisting}
> data Filesystem a = File a | Directory a [Filesystem a]
>	deriving(Show,Eq)
\end{lstlisting}

Para moverse dentro de esta estructura, se propone un data Breadcrumbs
que contenga el nombre del directorio donde se esta actualmente, seguido 
de una tripla que contienen listas. La primera lista indicará los 
directorios por lo que se ha bajado, es decir que son los padres del
directorio actual. La segunda lista es para mostrar los \texttt{Filesystem} que 
tiene a la izquierda del foco, y la tercera lista representa los 
\texttt{Filesystem} que están a la derecha del foco.

De esta manera tenemos lo siguiente: 


\begin{lstlisting}
> data Breadcrumbs a = 
>	Down a ([Filesystem a],[Filesystem a],[Filesystem a])
>		deriving(Show,Eq)
>
> type Zipper a = (Filesystem a, Breadcrumbs a)
\end{lstlisting}

Note que habrán movimientos que no se podrán hacer, y como no se sabe
que movimientos se haga, se trabajará con el tipo \texttt{Maybe}.

Con esta estructura queda ahora definir las funciones: 

\begin{lstlisting}
>
> goDown:: Zipper a -> Maybe (Zipper a)
> goDown  (File a,_) = Nothing
> goDown  (Directory a [],Down name ( lisDir ,_, _)) = Nothing
> goDown (Directory a xs,Down name ( lisDir ,y,ys)) = 
>    Just (head xs ,
>       Down a (Directory name 
>	      (reverse y++ [Directory a xs]++ys ):lisDir,
>	      [],
>	      tail xs))
>
> goRight:: Zipper a -> Maybe (Zipper a)
> goRight  (filesys, Down name ( lis, x, xs)) = 
>          if null xs then Nothing
>	   else Just (head xs ,Down name (lis,filesys:x,tail xs))
>
> goLeft:: Zipper a -> Maybe (Zipper a)
> goLeft (filesys, Down name ( lis, x, xs)) = 
>	if null x then Nothing
>	else Just (head x ,Down name (lis,tail x, filesys:xs))
>
> goBack:: Zipper a -> Maybe (Zipper a)
> goBack (_,Down name ( [] ,_,_)) = Nothing
> goBack (_,Down name ( lisDir ,_,_)) = 
>	Just (prilista (head lisDir),
>	      Down (nameDir (head lisDir)) 
>	      (tail lisDir,[],reslista (head lisDir)))
> 	 where prilista (Directory a b) = head b
>	       nameDir (Directory a b) = a
>	       reslista (Directory a b) = tail b
>
>	
> tothetop:: Zipper a -> Zipper a
> tothetop (tope,Down name ( [] ,lis,lis1)) =  
>		(tope,Down name ( [] ,lis,lis1))
> tothetop (tope,Down name ( lisdir,lis,lis1)) = 
>		tothetop $ 
>		fromJust $ 
>		goBack (tope,Down name ( lisdir,lis,lis1))
>
> modify:: ( a -> a ) -> Zipper a -> Zipper a
> modify f (File a ,Down name ( lisdir,lis,lis1)) = 
>		(File (f a) ,Down name ( lisdir,lis,lis1))
> modify f (Directory a b,Down name ( lisdir,lis,lis1)) = 
>		(Directory (f a) b ,Down name ( lisdir,lis,lis1))
>
> focus :: Filesystem a -> Zipper a
> focus (File a) = (File a,Down a ([],[],[]))
> focus (Directory a lis) = (Directory a lis ,Down a([] ,[],[]))
>
> defocus :: Zipper a -> Filesystem a
> defocus (File a, _ ) = File a
> defocus (Directory a lis,_) = Directory a lis
\end{lstlisting}

\end{document}
