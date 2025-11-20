import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class SecurityPacketRecord:
    """
    RTL/FPGA 쪽 SecurityPacket을 블록체인에 저장하는 단위.
    C 구조체:
        Bit#(128) pufID;
        Bit#(256) frameHash;
        Bit#(512) eccSignature;
        Bit#(64)  hwTimestamp;
        Bit#(32)  sequenceNumber;
        Bool      valid;
    """
    puf_id: str                 # pufID: 디바이스 고유 ID (hex string)
    frame_hash: str             # frameHash: SHA-256 해시 (hex string)
    ecc_signature: str          # eccSignature: ECC 서명 (r||s, hex string)
    hw_timestamp: int           # hwTimestamp: Unix timestamp (정수)
    sequence_number: int        # sequenceNumber: 단조 증가 카운터
    valid: bool                 # valid: 유효성 플래그
    device_id: str              # 추가: 디바이스/카메라 ID

    # 여기 meta에 video_id, frame_index, storage_path 등 넣어서
    # "영상 ↔ 해시 매핑"을 표현할 수 있음
    meta: Optional[Dict[str, Any]] = None


class Block:
    def __init__(
        self,
        index: int,
        timestamp: float,
        records: List[SecurityPacketRecord],
        previous_hash: str,
        nonce: int = 0,
        block_hash: Optional[str] = None,
    ):
        self.index = index
        self.timestamp = timestamp
        self.records = records
        self.previous_hash = previous_hash
        self.nonce = nonce
        # 해시값 들어오면 쓰고, 없으면 직접 생성해서 사용
        self.hash = block_hash or self.compute_hash()

    # 내부 정보를 담고 있는 hash 값 생성
    def compute_hash(self) -> str:
        # JSON 문자열로 직렬화하고, 그 위에 SHA-256 해시를 씌움
        block_dict = {
            "index": self.index,
            "timestamp": self.timestamp,
            "records": [asdict(r) for r in self.records],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        block_string = json.dumps(block_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(block_string).hexdigest()
    
    # 0 * difficulty가 될때까지 해시 채굴
    def mine(self, difficulty: int):
        target_prefix = "0" * difficulty
        while True:
            self.hash = self.compute_hash()
            if self.hash.startswith(target_prefix):
                print(
                    f"[BLOCKCHAIN] Block #{self.index} mined: "
                    f"nonce={self.nonce}, hash={self.hash[:18]}..."
                )
                break
            self.nonce += 1

    def to_dict(self) -> Dict[str, Any]:
        # JSON 직렬화용 딕셔너리로 변환. (Flask 응답, 피어 간 전송 시 사용)
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "records": [asdict(r) for r in self.records],
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Block":
        # JSON에서 역직렬화해서 Block 객체로 복원. (P2P로 받은 블록을 객체로 만들 때 사용)
        records = [SecurityPacketRecord(**r) for r in data["records"]]
        return cls(
            index=data["index"],
            timestamp=data["timestamp"],
            records=records,
            previous_hash=data["previous_hash"],
            nonce=data.get("nonce", 0),
            block_hash=data.get("hash"),
        )
