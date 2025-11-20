# sw/block_chain/server.py

import time
from typing import List, Optional, Dict, Any

from flask import Flask, request, jsonify

from .block import SecurityPacketRecord, Block
from .chain import Blockchain

app = Flask(__name__)

# 중앙에서 관리하는 단 하나의 블록체인
BLOCKCHAIN = Blockchain(difficulty=0)  # PoW 필요 없으면 0으로
PENDING_RECORDS: List[SecurityPacketRecord] = []


@app.post("/api/record")
def add_record():
    """
    AI / capture / hw 모듈에서 SecurityPacket을 보내는 엔드포인트.

    Body 예시:
    {
      "pufID": "abcd...",
      "frameHash": "0123...",
      "eccSignature": "deadbeef...",
      "hwTimestamp": 1730000000,
      "sequenceNumber": 42,
      "valid": true,
      "device_id": "cam01",
      "meta": {
        "video_id": "cam01_2025-11-21_10-00",
        "frame_index": 123,
        "storage_path": "/mnt/video/xxxx.mp4",
        "ai_result": {
            "deepfake_score": 0.02,
            "liveness_ok": true,
            "pass_flag": true
        }
      }
    }
    """
    data = request.get_json(force=True)

    def get_field(*names, default=None):
        for n in names:
            if n in data:
                return data[n]
        return default

    puf_id = get_field("puf_id", "pufID")
    frame_hash = get_field("frame_hash", "frameHash")
    ecc_signature = get_field("ecc_signature", "eccSignature")
    hw_timestamp = get_field("hw_timestamp", "hwTimestamp", default=int(time.time()))
    sequence_number = get_field("sequence_number", "sequenceNumber")
    valid = get_field("valid", default=True)
    device_id = data.get("device_id", "unknown-device")
    meta: Optional[Dict[str, Any]] = data.get("meta")

    missing = []
    if not puf_id:
        missing.append("pufID/puf_id")
    if not frame_hash:
        missing.append("frameHash/frame_hash")
    if not ecc_signature:
        missing.append("eccSignature/ecc_signature")
    if sequence_number is None:
        missing.append("sequenceNumber/sequence_number")

    if missing:
        return jsonify({"error": "missing fields", "missing": missing}), 400

    record = SecurityPacketRecord(
        puf_id=str(puf_id),
        frame_hash=str(frame_hash),
        ecc_signature=str(ecc_signature),
        hw_timestamp=int(hw_timestamp),
        sequence_number=int(sequence_number),
        valid=bool(valid),
        device_id=str(device_id),
        meta=meta,
    )

    # 1) 바로 블록에 넣고 싶으면:
    new_block = BLOCKCHAIN.add_block([record])

    # 2) 여러 개 모아서 만들고 싶으면:
    # PENDING_RECORDS.append(record)
    # if len(PENDING_RECORDS) >= 10:
    #     BLOCKCHAIN.add_block(PENDING_RECORDS[:])
    #     PENDING_RECORDS.clear()

    return jsonify({
        "status": "stored",
        "block_index": new_block.index,
        "block_hash": new_block.hash,
    }), 201


@app.get("/api/verify/<frame_hash>")
def verify(frame_hash: str):
    """
    허브 웹페이지나 외부 서비스에서
    '이 frame_hash가 실제 체인에 있는지' 확인하는 API.
    """
    result = BLOCKCHAIN.find_record_by_frame_hash(frame_hash)
    if not result:
        return jsonify({
            "exists": False,
            "message": "frame_hash not found in chain",
        }), 404

    return jsonify({
        "exists": True,
        "result": result,
    }), 200


@app.get("/api/blocks")
def list_blocks():
    """
    최근 블록들 조회 (대시보드용)
    ?limit=10 같은 식으로 조절 가능.
    """
    limit = int(request.args.get("limit", 10))
    chain_list = BLOCKCHAIN.to_list()
    return jsonify({
        "length": len(chain_list),
        "blocks": chain_list[-limit:],
    }), 200


@app.get("/api/status")
def status():
    """
    허브에서 간단히 상태 체크용:
    - 체인 길이
    - 제일 마지막 블록 시간
    """
    last = BLOCKCHAIN.last_block
    return jsonify({
        "chain_length": len(BLOCKCHAIN.chain),
        "last_block_index": last.index,
        "last_block_timestamp": last.timestamp,
    }), 200


if __name__ == "__main__":
    # 개발용 로컬 실행
    app.run(host="0.0.0.0", port=8000, debug=True)
