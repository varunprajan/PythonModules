import myio as Mio
from titlecase import titlecase
import copy

allreferences = ['batteries','compositemechanics','computationalfracture','constitutivemodels','dynamicfracture','hierarchicalbiocomposites','hybridcomposites','metallicglasses','misc','multiscale','notches','quasistaticfracture']

SPECIALCHARS = ['\'','^\\','"\\']
TAGNAMES = ['textit']

last_id = 0
            
def get_tag_escaped(line):
    braced = '{' in line
    if braced: # check if tag is legitimate (i.e. if brace is associated with '\\')
        idx = line.index('{')
        tagname = line[:idx]
        if len(tagname) > 1:
            braced = tagname in TAGNAMES
    if not braced:
        special = match_special(line)
        if special is not None:
            tagname = special
        else:
            tagname = ''
    return Tag(tagname,braced,0,True)
    
def get_tag_unescaped(line):
    return Tag('',True,0,False)

def match_special(line):
    for special in SPECIALCHARS:
        if line.startswith(special):
            return special
    return None
    
def exhaust_iterator(iterator,counter):
    for i in range(counter):
        next(iterator)
        
def update_tag_length(tags):
    for tag in tags:
        tag.length += 1
        
def get_different_tags(tagsold,tagsnew):
    differenttags = []
    for i, tagnew in enumerate(tagsnew):
        try:
            tagold = tagsold[i]
            if tagold != tagnew:
                differenttags.append(tagnew)
        except IndexError:
            differenttags.append(tagnew)
    return differenttags
          
def join_lists(listoflists,sep):
    res = listoflists[0].__class__()
    for list in listoflists:
        res.extend(list)
        res.append(TaggedChar(sep))
    res.pop()
    return res
    
class Tag(object):
    def __init__(self,name,braced,length,escaped):
        self.name = name
        self.braced = braced
        self.length = length
        self.escaped = escaped
        global last_id
        last_id += 1
        self.id = last_id
        
    def __eq__(self,other):
        return self.id == other.id
        
    def __str__(self):
        return 'Name: {0} \nBraced: {1}\nId: {2}'.format(self.name,self.braced,self.id)
    
    def gener_tagged_word(self,iterline):
        wordcounter = self.get_word_counter()
        i = 1
        while i <= wordcounter:
            charpossible = next(iterline)
            if isinstance(charpossible,str):
                i += 1
            yield charpossible
    
    def write_tagged_word(self,iterline):
        word = self.print_tag_begin()
        for taggedchar in self.gener_tagged_word(iterline):
            if isinstance(taggedchar,str):
                word += taggedchar
            else:
                word += taggedchar.write_tagged_word(iterline)
        word += self.print_tag_end()
        return word
    
    def print_tag_begin(self):
        pref, suff = '', ''
        if self.escaped:
            pref = '\\'
        if self.braced:
            suff = '{'
        return '{0}{1}{2}'.format(pref,self.name,suff)
        
    def print_tag_end(self):
        if self.braced:
            return '}'
        else:
            return ''

    def get_word_counter(self):
        '''Number of characters from start of word until end of word'''
        return self.length
            
    def get_tag_counter(self):
        '''Number of characters between first character in tag (either '\\' or '{') and first character that is tagged'''
        fudge = 0
        if self.braced and self.escaped:
            fudge = 1
        return len(self.name) + fudge
        
    def exhaust_tag(self,iterator):
        counter = self.get_tag_counter()
        exhaust_iterator(iterator,counter)
        
class TaggedChar(object):
    def __init__(self,char,tags=None):
        if tags is None:
            tags = []
        self.tags = tags
        self.char = char
        
    def add_deeper_tag(self,tag):
        self.tags.append(tag)
        
    def get_num_tags(self):
        return len(self.tags)
        
    def shorten_tags(self):
        for tag in self.tags:
            tag.length = 1

class TaggedLine(list):
    def __eq__(self,other):
        return self.get_char_list() == other.get_char_list()
    
    def parse_line(self,line):
        tags = []
        popnextstep = False
        iterline = enumerate(line)
        for i, char in iterline:
            if char in ['{','\\']:
                if char == '\\':
                    tag = get_tag_escaped(line[i+1:])
                elif char == '{':
                    tag = get_tag_unescaped(line[i+1:])
                tag.exhaust_tag(iterline)
                if not tag.braced:
                    popnextstep = True
                tags.append(tag)                
            elif char == '}':
                tags.pop()
            else:
                update_tag_length(tags)
                taggedchar = TaggedChar(char,copy.copy(tags)) # list is mutable
                self.append(taggedchar)
                if popnextstep:
                    tags.pop()
                    popnextstep = False
        
    def gen_seq_line(self):
        tagsold = []
        seqline = TaggedLineSeq()
        for taggedchar in self:
            tagsnew = taggedchar.tags
            differenttags = get_different_tags(tagsold,tagsnew)
            if differenttags:
                seqline.extend(differenttags)
            seqline.append(taggedchar.char)
            tagsold = tagsnew
        return seqline
    
    def title_line(self):
        capitalizedline = titlecase(self.write_tagless_line())
        for char, taggedchar in zip(capitalizedline,self):
            if taggedchar.get_num_tags() == 0:
                taggedchar.char = char
    
    def write_tagless_line(self):
        return ''.join(self.get_char_list())
        
    def write_tagged_line(self):
        seqline = self.gen_seq_line()
        return seqline.write_tagged_line()
        
    def get_char_list(self):
        return [taggedchar.char for taggedchar in self]
    
    def __str__(self):
        res = ''
        for taggedchar in self:
            tagstr = ';'.join([str(tag.id) for tag in taggedchar.tags])
            res += '{0}{1} \n'.format(taggedchar.char,tagstr)
        return res
        
    def strip_list(self,char):
        idxleft = 0
        for s in self:
            if (s.char == char):
                idxleft = idxleft + 1
            else:
                break
        idxright = len(self)
        for s in self[::-1]:
            if (s.char == char):
                idxright = idxright - 1
            else:
                break
        return self.__class__(self[idxleft:idxright])
        
    def strip_list_all(self,char):
        selfnew = self.__class__()
        for s in self:
            if (s.char != char):
                selfnew.append(s)
        return selfnew

    def split_list(self,char):
        charlist = self.get_char_list()
        if char not in charlist:
            return [self]
        else:
            idx = charlist.index(char)
            list1 = [self.__class__(self[:idx])]
            list2 = self.__class__(self[idx+1:])
            list1.extend(list2.split_list(char))
            return list1
            
    def create_initials(self):
        selfnew = TaggedLine()
        dotchar = TaggedChar('.')
        for taggedchar in self:
            taggedchar.shorten_tags()
            selfnew.append(taggedchar)
            selfnew.append(dotchar)
        return selfnew
            
class TaggedLineSeq(list):
    def write_tagged_line(self):
        linenew = ''
        iterself = iter(self)
        for char in iterself:
            if isinstance(char,str):
                linenew += char
            else:
                linenew += char.write_tagged_word(iterself)
        return linenew

class Title(object):
    def __init__(self,titleactual):
        self.titleactual = TaggedLine()
        self.titleactual.parse_line(titleactual)
        self.titlenew = copy.deepcopy(self.titleactual)
        self.titlenew.title_line()
    
    def __str__(self):
        return self.titlenew.write_tagless_line()

    def str_for_file(self):
        return self.titlenew.write_tagged_line()
    
class Author(object):
    def __init__(self,authoractual):
        self.authoractual = TaggedLine()
        self.authoractual.parse_line(authoractual)
        self.initials, self.last = self.get_author_print()
    
    def __eq__(self,other):
        if other.last.lower() == self.last.lower():
            if len(other.initials) == 1 or len(self.initials) == 1:
                return other.initials[0].char == self.initials[0].char
            else:
                return other.initials == self.initials               
        else:
            return False
    
    def __str__(self):
        return self.str_tagged()
    
    def str_tagless(self):
        strlast = self.last.write_tagless_line()
        if self.initials is not None:
            strinitials = self.initials.write_tagless_line()
            return '{1} {0}'.format(strlast,strinitials)
        else:
            return '{0}'.format(strlast)
        
    def get_author_print(self):
        first, last = self.parse_author()
        if first is not None:
            initials = self.parse_initials(first)
        else:
            initials = None
        return initials, last
        
    def parse_author(self):
        linenew = copy.deepcopy(self.authoractual)
        chars = linenew.get_char_list()
        if ',' in chars: # last name first
            last, first = linenew.split_list(',')
            last = last.strip_list(' ')
        elif ' ' in chars: # first name first
            names = linenew.split_list(' ')
            first = join_lists(names[:-1],' ')
            last = names[-1].strip_list(' ')
        else: # single name
            first = None
            last = self.authoractual
        return first, last
    
    def parse_initials(self,first):
        first = first.strip_list(' ')
        firstchars = first.get_char_list()
        if ' ' not in firstchars:
            if '.' not in firstchars: # single name
                initials = TaggedLine([first[0]])
            else: # initials not separated by spaces
                initials = first.strip_list_all('.')
        else: # several names or initials
            names = first.split_list(' ')
            initials = TaggedLine()
            for name in names:
                initials.append(name[0])
        return initials.create_initials()
            
    def str_tagged(self):
        strlast = self.last.write_tagged_line()
        if self.initials is not None:
            strinitials = self.initials.write_tagged_line()
            return '{0}, {1}'.format(strlast,strinitials)
        else:
            return '{0}'.format(strlast)
            
    def str_for_file():
        return self.str_tagged()
        
class AuthorList(list):    
    def __str__(self):
        return '; '.join([str(author) for author in self])
    
    def add_author(self,author):
        self.append(author)
       
    def new_authors_from_line(self,authorline):
        authorstrings = authorline.split(' and ')
        for authorstring in authorstrings:
            author = Author(authorstring)
            self.add_author(author)
            
    def str_for_file(self):
        return ' and '.join([author.str_for_file() for author in self])
    
class ReferenceList(list):
    def __init__(self):
        pass
    
    def add_reference(self,reference):
        self.append(reference)
    
    def add_reference2(self,key,type,**kwargs):
        newreference = Reference(key,type,**kwargs)
        self.add_reference(newreference) 
    
    def read_files_all(self,pref='references_',suff='.bib'):
        filenames = [pref + referencegroup + suff for referencegroup in allreferences]
        self.read_files(filenames)
    
    def read_files(self,filenames):
        for filename in filenames:
            self.read_file(filename)
    
    def read_file(self,filename):
        with open(filename,'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('@'):
                    type, key = self.get_type_key(line)
                    newreference = Reference(key,type)
                    newreference.read_file(f)
                    self.add_reference(newreference)
                    
    def write_file(self,filename):
        with open(filename,'w') as f:
            for reference in self:
                reference.write_file(f)
                    
    def get_type_key(self,line):
        articletype = Mio.findKey(line,'@','{')
        key = Mio.findKey(line,'{',',')
        return articletype, key
    
    def search_author(self,author):
        self._search_('authors',author)

    def search_title(self,keyword):
        self._search_('title',keyword)
        
    def _search_(self,field,keyword):
        counter = 1
        for reference in self:
            if keyword in getattr(reference,field):
                print('{0}) {1}'.format(counter,reference))
                counter = counter + 1

class Reference(object):
    def __init__(self,key,type,title=None,authors=None,pages=None,year=None,journal=None,number=None,volume=None,**kwargs):
        self.key = key
        self.type = type
        self.title = title
        self.authors = authors
        self.pages = pages
        self.year = year
        self.journal = journal
        self.number = number
        self.volume = volume

    def __str__(self):
        str1 = 'Key: {0} \n'.format(self.key)
        str2 = 'Title: {0} \n'.format(self.title)
        str3 = 'Authors: {0}'.format(self.authors)
        return str1 + str2 + str3
        
    def read_file(self,f):
        blockoflines = Mio.generReadUntil(f,'}')
        for line in blockoflines:
            key = Mio.findKey(line,'','=').strip()
            value = Mio.findKey(line,'{','}').strip()
            if key == 'author':
                authors = AuthorList()
                authors.new_authors_from_line(value)
                self.authors = authors
            else:
                if key == 'title':
                    value = Title(value)
                setattr(self,key,value)
                
    def write_file(self,f):
        for key, val in self.__dict__.items():
            try:
                betweenbraces = val.str_for_file()
            except AttributeError:
                betweenbraces = val
            f.write('{0} = {{1}}'.format(key,betweenbraces))
    
            