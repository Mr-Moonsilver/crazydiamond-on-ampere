"""Serial synthetic sparse row-run planner; no engine integration or GPU import."""
from dataclasses import dataclass

PAGE_ROWS=4048
ROW_BYTES=1024
ARENA_PAGES=24
PIECE_PAGES=4
MAX_PIECES=256


@dataclass(frozen=True)
class Piece:
    source_row: int
    arena_row: int
    rows: int


@dataclass(frozen=True)
class Bucket:
    compact_start: int
    rows: int
    pieces: tuple


@dataclass(frozen=True)
class Plan:
    epoch: int
    mapping: tuple
    versions: tuple
    selected: tuple
    buckets: tuple
    fallback: str | None


def make_plan(selected, mapping, versions, epoch, *, page_rows=PAGE_ROWS,
              arena_pages=ARENA_PAGES, max_pieces=MAX_PIECES):
    """Preserve compact order; coalesce only physically adjacent source rows.

    Every piece spans at most four physical pages AND at most four pages of
    bytes. No plan is admitted partially: a fragmented bucket refuses the
    whole selection before any GPU work. Epoch/version checks are synthetic
    pre-submit guards, not real engine scheduler leases.
    """
    selected, mapping, versions=tuple(selected),tuple(mapping),tuple(versions)
    if (type(epoch) is not int or epoch<1 or type(page_rows) is not int or page_rows<1
            or type(arena_pages) is not int or arena_pages<1
            or type(max_pieces) is not int or max_pieces<1):
        raise ValueError('invalid_geometry_or_epoch')
    if (not versions or any(type(v) is not int or v<0 for v in versions)
            or any(type(p) is not int or not 0<=p<len(versions) for p in mapping)):
        raise ValueError('invalid_source_map')
    if any(type(t) is not int or t<0 or t>=len(mapping)*page_rows for t in selected):
        raise ValueError('invalid_selection')
    if any(a>=b for a,b in zip(selected,selected[1:])):
        raise ValueError('selection_must_be_sorted_unique_without_padding')
    buckets=[]
    for lo in range(0,len(selected),arena_pages*page_rows):
        part=selected[lo:lo+arena_pages*page_rows]
        pieces=[]
        for dst,t in enumerate(part):
            src=mapping[t//page_rows]*page_rows+t%page_rows
            if (pieces and pieces[-1].source_row+pieces[-1].rows==src
                    and pieces[-1].arena_row+pieces[-1].rows==dst
                    and pieces[-1].rows<PIECE_PAGES*page_rows
                    and src//page_rows-pieces[-1].source_row//page_rows<PIECE_PAGES):
                p=pieces[-1];pieces[-1]=Piece(p.source_row,p.arena_row,p.rows+1)
            else:
                pieces.append(Piece(src,dst,1))
                if len(pieces)>max_pieces:
                    return Plan(epoch,mapping,versions,selected,(),'descriptor_cap')
        buckets.append(Bucket(lo,len(part),tuple(pieces)))
    return Plan(epoch,mapping,versions,selected,tuple(buckets),None)


def validate_source(plan, mapping, versions, epoch):
    mapping, versions = tuple(mapping), tuple(versions)
    if (not isinstance(plan, Plan) or type(epoch) is not int
            or any(type(x) is not int for x in (*mapping, *versions))
            or plan.epoch!=epoch or plan.mapping!=tuple(mapping) or plan.versions!=tuple(versions)):
        raise ValueError('stale_source_or_mapping')
    if plan.fallback:
        raise ValueError('not_dma_admitted')


def cases():
    p=PAGE_ROWS
    identity=tuple(range(50))
    fragmented=tuple((i*17+3)%52 for i in range(50))
    return [
        ('empty',(),identity,False),
        ('one_row',(p+17,),fragmented,False),
        ('page_edges',(0,p-1,p,p+1,2*p-1),fragmented,False),
        ('four_page_piece_cap',tuple(range(8*p)),identity,False),
        ('unaligned_four_page_piece_cap',tuple(range(1,8*p+3)),identity,False),
        ('arena_23_pages',tuple(range(23*p)),fragmented,False),
        ('arena_24_pages',tuple(range(24*p)),fragmented,False),
        ('arena_25_pages_tail',tuple(range(25*p+13)),fragmented,False),
        ('bounded_subpage_runs',tuple(t for i in range(128) for t in range(i*64,i*64+32)),fragmented,False),
        ('aliased_physical_pages',tuple(range(2*p)),(3,3)+identity[2:],False),
        ('near_sparse_threshold',tuple(range(40*p-1)),fragmented,False),
        ('fragmentation_fallback',tuple(range(0,1024,2)),fragmented,True),
    ]
