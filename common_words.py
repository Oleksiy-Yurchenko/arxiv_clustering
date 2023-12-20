common = ['use', 'arxiv', 'circa', 'result', 'model', 'show', 'using', 'study', 'system', 'paper', 'present', 'st',
          'nd', 'rd', 'such', 'reply', 'comment', 'remark', 'work', 'author', 'publication', 'note', 'review',
          'overview', 'reader', 'determine', 'give', 'experiment', 'article', 'problem', 'discuss', 'address',
          'compute', 'submission', 'examine', 'proof', 'rejoinder', 'citation', 'abstract', 'title', 'versa', 'versus',
          'vice', 'whereas', 'passim', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'ten', '20th', 'eighth', '19th', '21st',
          'fifth', 'ity', 'ing', 'ings', 'ible', 'bo', 'rev', '5th', '4th', 'nineteenth', 'meter', 'decimeter', '17th',
          '60th', 'thatvis', 'thatv', '18th', '2nd', '1st', 'whatsoever', '100th', '30th', '7th', '50th', '70th',
          'thirteenth', '13th', 'foreword', 'fifteenth', '80th', '25th', 'appendix', 'cited', 'quoted', 'viceversa',
          'type', 'set', 'trations', 'tration', 'pesquisas', 'pesquisa', 'thenτis', 'thenτ', 'enteros', 'entero',
          'segundas', 'segunda', 'conceptos', 'concepto', 'fuentes', 'fuente', 'speculationes', 'astrophy', 'astroph',
          'approxcs', 'approx', 'muchos', 'mucho', 'whereass', 'wherea', 'uuencoded', 'postscript']


withdrawn = ['withdraw', 'withdrew', 'withdrawn', 'withdrown', 'withdrow', 'wihdrawn']

abbr_list = ['ad', 'anon', 'c', 'ca', 'ce', 'cf', 'cit', 'def', 'eg', 'ed',
             'et', 'eds', 'el', 'id', 'i', 'ii', 'iii', 'ie', 'ibid', 'inf',
             'etc', 'illus', 'loc', 'ms.', 'ms', 'mss', 'na', 'nb', 'nd', 'no',
             'op', 'p', 'pg', 'pp', 'pgs', 'pseud', 'pub', 'qtd', 'qv', 'resp',
             's', 'sic', 'sc', 'sca', 'sce', 'sup', 'trans', 'up', 'v', 'viz', 'vol',
             'vols', 'vs']

greek = ['ρ', 'ghz', 'θ', 'nm', 'ε', 'η', 'ψ', 'dm', 'R', 'Z', 'κ', 'sd', 'ξ',
         'χ', 'C', 'ac', 'dc', 'sr', 'F', 'ml', 'N', 'mhz', 'le', 'O', 'μm',
         'http', '∂', 'em', 'H', 'Q', 'gr', 'M', 'A', 'ch', 'B', 'P', 'ζ',
         'λcdm', 'ln', 'phys', 'ph', '∑', 'D', 'G', 'S', 'L', 'oh', 'T', 'mk',
         'E', 'rt', 'aa', 'us', 'km', 'cm', 'mm', 'γγ', 'X', 'ij', 'K', 'cs',
         'www', 'ary', 'gj', 'dy', 'ao', 'pi0', 'id', 'dn', 'ππ', 'ho', 'fo',
         'po', 'uk', 'cg', 'av', 'ds', 'U', '∂ω', 'ro', 'ea', 'ps', 'qc', 'bm',
         'V', 'jc', 'W', 'nt', 'PT', 'ue', 'ou', 'yu', 'μs', 'kv', 'lν', 'ka',
         'kα', 'vp', 'νν', 'oe', 'ł', 'J', 'δφ', 'kw', 'tu', 'dk', 'qi', 'sγ',
         'dμ', 'oo', 'ie', 'zh', 'λλ', 'cj', 'kπ', 'Y', 'δγ', 'kh', 'eγ', 'μg',
         'ty', 'dσ', 'gu', 'zw', 'ħω', 'bu', 'δν', 'μτ', 'bj', 'ι', 'μk', 'kj',
         'ji', 'cm3', 'm2', 'gy', 'αβ', 'dη', 'ms', 'ke', 'vw', 'ττ', 'zγ',
         'ψφ', 'jr', 'δρ', 'ny', 'mmm', 'δα', 'NP', 'γη', 'nπ', 'δn', 'eν',
         'dπ', 'eμ', 'iy', 'ay', 'πς', '∇φ', 'pj', 'rj', 'nδ', 'μhz', 'δv',
         'ρπ', 'aq', 'mah', 'gμ', 'μw', 'μv', 'ug', 'fσ', 'ψψ', 'nj', 'μγ',
         'sinθ', 'sin', 'cos', 'tg', 'arctg', 'arcctg', 'aj', 'tz', 'vu', 'dγ',
         'ry', 'δm', 'δω', 'δu', 'i∂', 'δη', 'μas', 'ya', 'fj', 'qu', 'kc',
         'ββ', 'δμ', 'ur', 'δλ', '℘', 'δt', 'ki', 'kω', 'iω', 'qo', 'ρρ', 'χχ',
         'a∩', 'yo', 'cγ', 'bbb', 'hy', 'μa', 'ηη', 'γχ', 'δθ', 'dφ', 'uz',
         'δlog', 'logl', '∂∂', 'eτ', 'þ', 'dρ', 'πn', 'δδ', 'pγ', 'yi', 'CP',
         'yh', 'dω', 'δs', '∂σ', 'N∪', 'dτ', 'δc', 'aγ', 'πη', 'logσ', 'cosφ',
         'pλ', 'γp', 'iz', 'dλ', 'SL', 'ψγ', 'γis', 'δd', 'δε', 'ργ', 'βγ',
         'jf', 'πγ', 'dξ', 'log10', 'iγ', 'mω', 'δg', 'zg', 'jz', 'ς', '∂φ',
         'pZ', 'nω', 'αα', 'zm', 'δh', 'nφ', 'zzz', 'ωτ', 'vγ', 'a∪', 'iθ',
         'logλ', 'δπ', 'km3', 'lao0', 'n∑', 'ωφ', '∂ν', 'hρ', 'δκ', 'λγ', 'u∇',
         'λφ', 'ηγ', 'δb', 'ηπ', 'hν', 'i∩', 'bπ', 'nλ', 'a∇', 'qy', 'dν', 'ψη',
         'μt', 'λμ', 'δf', 'jo', 'cλ', 'iδ', 'dθ', 'ψk', 'δz', 'χ∇', 'dς', '∂D',
         '∇u', 'iα', 'pπ', 'πδ', 'tγ', 'δp', 'nσ', 'μn', 'zu', 'ωρ', 'δa', 'ψπ',
         'sπ', 'iλ', 'p∣', 'μφ', 'iσ', 'μc', 'φγ', 'cosδ', 'μω', 'zi', 'iφ',
         'βδ', 'yf', 'πν', 'mZ', 'cφ', 'kς', '∂u', 'ωχ', 'Zg', 'δy', 'δq', 'nγ',
         'kδ', 'SO', 'NN', 'Lu', 'inω', 'Set', 'SU', 'tδ', 'R∪', 'auf', 'λf',
         'f∩', 'on∂ω', 'γνν', '∇ρ', 'pφ', 'αlog', 'ertion', 'PA', 'Sp', '∂m',
         'BR', '∑m', 'ß', 'εlog', 'ψdm', 'k∩', 'iħ', 'nθ', '∩ω', 'GP', 'ƪ',
         'uZ', 'pβ', '∂f', 'iμ', 'πħ', 'iness', 'ZFC', 'nε', '∂ρ', 'uF', 'Gr',
         'p∩', '0', 's∪', 'e∩', 'πα', 'kφ', 'nμ', 'AD', 'αg', 'Cp', 'νφ', '∂δ',
         'νββ', 'PR', 'γδ', 'λ0', 'bδ', 'iti', 'icis', 'esas', 'lnδ', 'σς',
         'λe', 'bφ', 'g⋉', 'GI', 'aδ', 'u∂', 'Rf', 'i∇', 'λα', '°c', 'hF', 'iς',
         'h∩', 'Lie', 'PL', '∂γ', '∇ψ', '∂ln', 'sφ', 'Rep', 'ad', 'br', 'cp',
         'gi', 'gp', 'hf', 'lie', 'lu', 'mz', 'nn', 'np', 'on', 'pa', 'pl',
         'pr', 'pt', 'pz', 'rep', 'rf', 'set', 'sl', 'so', 'sp', 'su', 'uf', 'zfc' ]