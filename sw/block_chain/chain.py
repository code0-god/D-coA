import time
from typing import List, Dict, Any, Optional

from block import Block, SecurityPacketRecord


class Blockchain:
    def __init__(self, difficulty: int = 3):
        self.difficulty = difficulty
        self.chain: List[Block] = []
        self.create_genesis_block()

    # 0번 블록 (제네시스 블록)
    def create_genesis_block(self):
        genesis_record = SecurityPacketRecord(
            puf_id="GENESIS_PUF",
            frame_hash="GENESIS_FRAME",
            ecc_signature="GENESIS_SIG",
            hw_timestamp=int(time.time()),
            sequence_number=0,
            valid=True,
            device_id="SYSTEM",
            meta={"info": "Genesis block"},
        )
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            records=[genesis_record],
            previous_hash="0",  # 앞 블록이 없으니까 0으로 고정
        )
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)

    # @property -> 외부에서는 () 없이 blockchain.last_block로 접근
    @property  
    def last_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, records: List[SecurityPacketRecord]) -> Block:
        # 주어진 SecurityPacketRecord 리스트를 새 블록에 담아서 mine을 수행한 뒤 체인에 추가.
        new_block = Block(
            index=self.last_block.index + 1,
            timestamp=time.time(),
            records=records,
            previous_hash=self.last_block.hash,
        )
        new_block.mine(self.difficulty)
        self.chain.append(new_block)
        return new_block

    # 체인 유효성 검사
    def is_valid(self, chain: Optional[List[Block]] = None) -> bool:
        chain = chain or self.chain
        for i in range(1, len(chain)):
            prev = chain[i - 1]
            curr = chain[i]

            if curr.previous_hash != prev.hash:
                print(f"[ERROR] Block #{curr.index}: previous_hash mismatch")
                return False

            if curr.hash != curr.compute_hash():
                print(f"[ERROR] Block #{curr.index}: hash invalid (tampered?)")
                return False

        return True

    def to_list(self) -> List[Dict[str, Any]]:
        # 체인을 JSON 직렬화용 리스트(dict 리스트)로 변환. (HTTP 응답, P2P 송수신 시 사용)
        return [b.to_dict() for b in self.chain]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]], difficulty: int = 3) -> "Blockchain":
        # JSON 리스트를 받아서 Blockchain 객체로 복원.
        blockchain = cls(difficulty=difficulty)
        blockchain.chain = [Block.from_dict(b) for b in data]
        return blockchain

    # 외부에서 받은 새 체인으로 교체.
    def replace_chain(self, new_blocks: List[Block]) -> bool:
        if not self.is_valid(new_blocks):
            return False
        if len(new_blocks) <= len(self.chain):
            return False

        self.chain = new_blocks
        print(f"[BLOCKCHAIN] Chain replaced with new chain of length {len(new_blocks)}")
        return True
    
    # 특정 frame_hash를 가진 SecurityPacketRecord를 검색
    def find_record_by_frame_hash(self, frame_hash: str) -> Optional[Dict[str, Any]]:
        for block in self.chain:
            for rec in block.records:
                if rec.frame_hash == frame_hash:
                    return {
                        "block_index": block.index,
                        "block_hash": block.hash,
                        "record": rec.__dict__,  # asdict(rec) 써도 됨
                    }
        return None
