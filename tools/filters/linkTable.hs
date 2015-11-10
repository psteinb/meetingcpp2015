import Text.Pandoc.JSON
import Text.Pandoc.Walk
import Data.Monoid

-- http://stackoverflow.com/questions/26406816/pandoc-is-there-a-way-to-include-an-appendix-of-links-in-a-pdf-from-markdown/26415375#26415375

main :: IO ()
main = toJSONFilter appendLinkTable

appendLinkTable :: Pandoc -> Pandoc
appendLinkTable (Pandoc m bs) = Pandoc m (bs ++ linkTable bs)

linkTable :: [Block] -> [Block]
linkTable p = [Header 2 ("linkTable", [], []) [Str "Links"] , Para links]
  where
    links = concatMap makeRow $ query getLink p
    getLink (Link txt (url, _)) = [(url,txt)]
    getLink _ = []
    makeRow (url, txt) = txt ++ [Str ":", Space, Link [Str url] (url, ""), LineBreak]
