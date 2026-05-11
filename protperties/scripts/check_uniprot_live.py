import logging
from protperties.io_uniprot import fetch_uniprot_sequences

logging.basicConfig(level=logging.INFO)

def main():
    # Mix of common IDs, accessions that previously failed, and isoforms
    accessions = [
        "O00115", "O00116", "O00139", # IDs that failed before
        "P04637", "Q9Y6K9", # Common p53, NEMO
        "P38398", "Q15149", "Q99708", # More IDs
        "O00115-2", "A0AV96-2", # Isoforms
        "A0A1B0GTW7", "A0A2R8Y3T5", # TrEMBL and secondary
        "Q00653", "Q01196", "Q01484", 
        "Q01844", "Q01860", "Q02078", 
        "Q02223", "Q02241"
    ]
    
    print(f"Testing {len(accessions)} accessions...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_cache:
        results = fetch_uniprot_sequences(accessions, cache_dir=tmp_cache)
    
    print(f"\nFetched {len(results)} sequences out of {len(accessions)} requested.")
    
    for acc in accessions:
        if acc in results:
            print(f"[OK] {acc} -> length {len(results[acc])}")
        else:
            print(f"[MISSING] {acc}")

if __name__ == "__main__":
    main()
