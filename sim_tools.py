
import json

def json_append(filename, entry):
    '''
    This function incrementally add entries to a json file
    while keeping the format correct

    Parameters
    ----------
    filename: str
        the name of the JSON file
    entry:
        the new entry to append
    '''

    with open(filename, 'at') as f:

        if f.tell() == 0:
            # first write, add array
            json.dump([entry], f, indent=0)

        else:
            # remove last character ']' and '\n'
            f.seek(f.tell() - 2, 0)
            f.truncate()

            # add missing comma to previous element
            f.write(',\n')

            # dump the latest entry
            json.dump(entry, f, indent=0)

            # close the json file
            f.write('\n]')

