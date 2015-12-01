#!/usr/bin/env python
from pandocfilters import toJSONFilter
from pandocfilters import stringify


link_list = {}

def my_parse(k, v, fmt, meta):
    #global link_list
    if k == 'Link':
        link_list[stringify(v)] = v[-1][0]


if __name__ == "__main__":
    toJSONFilter(my_parse,None)

    count = 0
    pages = 1
    sorted_keys = sorted(link_list.keys())
    widest_key = max([ len(item) for item in sorted_keys ])
    widest_url = max([ len(item) for item in link_list.values() ])
    
    print "## Links"
    on_one_page = 8
    n_pages = (len(sorted_keys)+on_one_page-1)/on_one_page

    print widest_key*"-",widest_url*"-"
    for key  in sorted_keys:
        if count and count % on_one_page == 0:
            print widest_key*"-",widest_url*"-"
            print "\n\n## Links %i/%i\n" % (pages+1,n_pages)
            print widest_key*"-",widest_url*"-"
            pages += 1
        print ("%"+str(widest_key)+"s %"+str(widest_url)+"s") % (key,link_list[key])
        count += 1
    print widest_key*"-",widest_url*"-"
