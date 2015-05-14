import Data.List
import Data.Functor
import Data.Monoid
import Data.Foldable (foldMap)
import Data.Tree
import Data.Maybe (fromJust)



data Sample a = Sample { x :: [a], y :: a }
      deriving (Show)
      
data Hypothesis a = Hypothesis { c :: [a] }
	deriving (Show)
	
alpha :: Double
alpha = 0.03

epsilon :: Double
epsilon = 0.0000001

guess :: Hypothesis Double
guess = Hypothesis { c = [0.0, 0.0, 0.0] }

training :: [Sample Double]
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




-- ---------------------- MACHINE LEARNING-----------------------------------
  
veryClose :: Double -> Double -> Bool
veryClose a b = abs (a - b) <= epsilon 

addOnes :: [Sample Double] -> [Sample Double]
addOnes = map agregar 
  where agregar m = Sample {x = add (x m), y = y m}
	add lis =  1:lis
	
	
theta :: Hypothesis Double -> Sample Double -> Double
theta h s = foldl' (+) 0 ( zipWith (*) (c h) (x s))


cost :: Hypothesis Double -> [Sample Double] -> Double
cost h ss = (foldr ((+).eval h) 0 ss) / (fromIntegral (length ss)*2)
  where eval hy sam = (theta hy sam - y sam)^2
	
descend :: Double -> Hypothesis Double -> [Sample Double] -> Hypothesis Double
descend alpha h ss = Hypothesis {c = reverse ( fst ( foldl' (calculo (length ss) alpha h ss ) ([],0) (c h)))}
  where calculo tam alpha hs ss (lis, num) h = (sumatoria tam alpha ss hs num : lis, num+1) 
	sumatoria tam alpha ss hy num = c hy !! num - ((foldr ((+).funaux hy num) 0 ss) * alpha/ fromIntegral tam) 
	funaux hyp num s = ((theta hyp s) - y s)* (x s !! num)



gd :: Double -> Hypothesis Double -> [Sample Double] -> [(Integer,Hypothesis Double,Double)]
gd alpha h ss = unfoldr (\(i,hypo,c) -> 
			    if veryClose c (cost (descend alpha hypo (addOnes ss)) (addOnes ss)) then Nothing
			    else Just ((i,hypo,c),calculo i alpha hypo (addOnes ss))) (0,h,cost h (addOnes ss))
  where calculo i alpha hy ss = (i+1,descend alpha hy ss, cost (descend alpha hy ss) ss)

						       

-- ---------------- -------MONOIDE------------------------

newtype Max a = Max { getMax :: Maybe a }
	deriving ( Eq ,Ord,Show)
	
instance (Ord a) => Monoid ( Max a ) where
	mempty                       = Max Nothing
	mappend (Max x) (Max y) = Max $ max x y
	
	
	
-- ---------------- ZIPPER -----------------------------------

data Filesystem a = File a | Directory a [ Filesystem a ]
      deriving(Show,Eq)

      
    
data Breadcrumbs a = Down a ([Filesystem a],[Filesystem a],[Filesystem a])
	      deriving(Show,Eq)

type Zipper a = ( Filesystem a , Breadcrumbs a )
      
testFile :: Filesystem Integer
testFile = Directory 2 [File 3,File 4,Directory 5 [File 6],File 7]

testFile2 :: Filesystem Integer
testFile2 = Directory 2 [File 3,File 4,Directory 5 [File 6,Directory 8 [File 9,File 10]],File 7]


testFile1 :: Filesystem Integer
testFile1 = File 42

testFile3 :: Filesystem Integer
testFile3 = Directory 42 []

goDown:: Zipper a -> Maybe (Zipper a)
goDown  (File a,_) = Nothing
goDown  (Directory a [],Down name ( lisDir ,_, _)) = Nothing
goDown (Directory a xs,Down name ( lisDir ,y,ys)) = 
		      Just (head xs ,Down a (Directory name (reverse y++ [Directory a xs]++ys ):lisDir ,[],tail xs))

		      

goRight:: Zipper a -> Maybe (Zipper a)
goRight  (filesys, Down name ( lis, x, xs)) = 
			    if null xs then Nothing
				else Just (head xs ,Down name (lis,filesys:x,tail xs))

goLeft:: Zipper a -> Maybe (Zipper a)
goLeft (filesys, Down name ( lis, x, xs)) = 
			    if null x then Nothing
				      else Just (head x ,Down name (lis,tail x, filesys:xs))
						 
goBack:: Zipper a -> Maybe (Zipper a)
goBack (_,Down name ( [] ,_,_)) = Nothing
goBack (_,Down name ( lisDir ,_,_)) = Just (prilista (head lisDir),Down (nameDir (head lisDir)) (tail lisDir,[],reslista (head lisDir)))
  where prilista (Directory a b) = head b
	nameDir (Directory a b) = a
	reslista (Directory a b) = tail b

tothetop:: Zipper a -> Zipper a
tothetop (tope,Down name ( [] ,lis,lis1)) =  (tope,Down name ( [] ,lis,lis1))
tothetop (tope,Down name ( lisdir,lis,lis1)) = tothetop $ fromJust $ goBack (tope,Down name ( lisdir,lis,lis1))

modify:: ( a -> a ) -> Zipper a -> Zipper a
modify f (File a ,Down name ( lisdir,lis,lis1)) = (File (f a) ,Down name ( lisdir,lis,lis1))
modify f (Directory a b,Down name ( lisdir,lis,lis1)) = (Directory (f a) b ,Down name ( lisdir,lis,lis1))

focus :: Filesystem a -> Zipper a
focus (File a) = (File a,Down a ([],[],[]))
focus (Directory a lis) = (Directory a lis ,Down a([] ,[],[]))

defocus :: Zipper a -> Filesystem a
defocus (File a, _ ) = File a
defocus (Directory a lis,_) = Directory a lis








