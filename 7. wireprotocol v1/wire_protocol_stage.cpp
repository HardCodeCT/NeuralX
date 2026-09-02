// ============================================================================
// wire_protocol_stage.cpp
//
// Excerpt: the actual framing + TCP send/recv loop code from
// wire_protocol.h, sender_client.cpp, and receiver_server.cpp, unmodified.
// This is what "why TCP" and "how frames are framed" refer to in the post.
// ============================================================================

// ---------------------------------------------------------------------------
// wire_protocol.h (full file, shared verbatim by both sender and receiver
// via #include -- this is what guarantees the two sides can't drift apart
// on the message format)
// ---------------------------------------------------------------------------
#pragma once
#include <cstdint>
#include <vector>
#include <cstring>

namespace wire {

enum MsgType : uint8_t {
    MSG_CONFIG      = 1,  // payload: fullW(4) fullH(4) lowW(4) lowH(4) blockSize(4) blocksX(4) blocksY(4)
    MSG_BASE        = 2,  // payload: one MJPEG-encoded low-res frame (full-frame mode)
    MSG_ENH         = 3,  // payload: one FFV1-encoded full-res residual frame (full-frame mode)
    MSG_TILE        = 4,  // unused by the current pipeline; reserved
    MSG_FRAME_FULL  = 5,  // payload: empty; marks frame_index as full-frame mode, BASE+ENH follow
    MSG_FRAME_DELTA = 6,  // payload: rect_x,rect_y,rect_w,rect_h + LZ4-compressed rect bytes
    MSG_END         = 7,  // payload: empty
};

constexpr uint32_t NO_TILE = 0xFFFFFFFFu;

inline void putU32(std::vector<uint8_t> &b, uint32_t v) {
    b.push_back(uint8_t((v >> 24) & 0xFF)); b.push_back(uint8_t((v >> 16) & 0xFF));
    b.push_back(uint8_t((v >> 8) & 0xFF));  b.push_back(uint8_t(v & 0xFF));
}
inline uint32_t getU32(const uint8_t *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

// Fixed 17-byte header: type(1) + frame_index(8) + tile_index(4) + payload_size(4).
// Every message on the wire starts with exactly this, big-endian throughout.
struct Header {
    uint8_t type;
    int64_t frameIndex;
    uint32_t tileIndex = NO_TILE;
    uint32_t payloadSize;

    void serialize(uint8_t out[17]) const {
        out[0] = type;
        for (int i = 0; i < 8; ++i)
            out[1 + i] = uint8_t((uint64_t(frameIndex) >> ((7 - i) * 8)) & 0xFF);
        out[9]  = uint8_t((tileIndex >> 24) & 0xFF);
        out[10] = uint8_t((tileIndex >> 16) & 0xFF);
        out[11] = uint8_t((tileIndex >> 8) & 0xFF);
        out[12] = uint8_t(tileIndex & 0xFF);
        out[13] = uint8_t((payloadSize >> 24) & 0xFF);
        out[14] = uint8_t((payloadSize >> 16) & 0xFF);
        out[15] = uint8_t((payloadSize >> 8) & 0xFF);
        out[16] = uint8_t(payloadSize & 0xFF);
    }
    static Header parse(const uint8_t in[17]) {
        Header h;
        h.type = in[0];
        uint64_t fi = 0;
        for (int i = 0; i < 8; ++i) fi = (fi << 8) | in[1 + i];
        h.frameIndex = int64_t(fi);
        h.tileIndex = getU32(in + 9);
        h.payloadSize = getU32(in + 13);
        return h;
    }
};

constexpr int HEADER_SIZE = 17;

// ---- FRAME_DELTA payload: rect header (16 bytes) + LZ4-compressed rect
// pixel bytes. rect_w/rect_h == 0 means "unchanged, no rect follows". ----
constexpr size_t RECT_HEADER_SIZE = 16; // 4 x uint32

inline std::vector<uint8_t> buildEmptyDeltaPayload() {
    std::vector<uint8_t> b;
    putU32(b, 0); putU32(b, 0); putU32(b, 0); putU32(b, 0); // x,y,w=0,h=0
    return b;
}
inline void buildDeltaPayloadHeader(std::vector<uint8_t> &out, int rectX, int rectY, int rectW, int rectH) {
    out.clear();
    putU32(out, uint32_t(rectX)); putU32(out, uint32_t(rectY));
    putU32(out, uint32_t(rectW)); putU32(out, uint32_t(rectH));
    // caller appends the LZ4-compressed bytes after this call
}

} // namespace wire

// ---- Hardcoded endpoint. Both sender and receiver include this same
// header, so the two stay in sync automatically. ----
namespace endpoint {
    constexpr const char *HOST = "127.0.0.1";
    constexpr uint16_t    PORT = 5000;
}

// ---------------------------------------------------------------------------
// Sender side: socket setup + sendAll loop (from sender_client.cpp)
// ---------------------------------------------------------------------------
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdexcept>

class TcpSender {
public:
    void connectTo(const char *host, uint16_t port) {
        sock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, host, &addr.sin_addr);

        if (connect(sock_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
            throw std::runtime_error("connect() failed");

        // Disable Nagle's algorithm -- without this, small header-only
        // messages (MSG_FRAME_FULL, empty deltas) can sit buffered for up
        // to ~40ms waiting to coalesce with more data before the OS sends
        // them, which defeats the point of pacing frames at 24fps.
        int one = 1;
        setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof(one));
    }

    void sendMessage(wire::MsgType type, int64_t frameIndex, const uint8_t *data, uint32_t size,
                      uint32_t tileIndex = wire::NO_TILE) {
        wire::Header h;
        h.type = static_cast<uint8_t>(type);
        h.frameIndex = frameIndex;
        h.tileIndex = tileIndex;
        h.payloadSize = size;
        uint8_t hdr[wire::HEADER_SIZE];
        h.serialize(hdr);
        sendAll(hdr, sizeof(hdr));
        if (size) sendAll(data, size);
    }
    void sendMessage(wire::MsgType type, int64_t frameIndex, const std::vector<uint8_t> &data,
                      uint32_t tileIndex = wire::NO_TILE) {
        sendMessage(type, frameIndex, data.empty() ? nullptr : data.data(),
                    static_cast<uint32_t>(data.size()), tileIndex);
    }

private:
    SOCKET sock_ = INVALID_SOCKET;

    // TCP is a byte stream: send() has no obligation to send all `size`
    // bytes in one call (partial sends are legal, especially once the
    // socket send buffer fills). This loop keeps calling send() with
    // whatever's left until the whole message is actually on the wire.
    void sendAll(const uint8_t *data, size_t size) {
        size_t sent = 0;
        while (sent < size) {
            int rc = send(sock_, reinterpret_cast<const char *>(data) + sent,
                           static_cast<int>(size - sent), 0);
            if (rc <= 0) throw std::runtime_error("send() failed");
            sent += static_cast<size_t>(rc);
        }
    }
};

// ---------------------------------------------------------------------------
// Receiver side: socket setup + recvAll/recvMessage loop (from
// receiver_server.cpp)
// ---------------------------------------------------------------------------
class TcpReceiver {
public:
    void listenAndAccept(uint16_t port) {
        listenSock_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

        int one = 1;
        setsockopt(listenSock_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&one), sizeof(one));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);
        bind(listenSock_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr));

        if (listen(listenSock_, 1) != 0) // backlog of 1: single-connection tool by design
            throw std::runtime_error("listen() failed");

        client_ = accept(listenSock_, nullptr, nullptr);
        if (client_ == INVALID_SOCKET) throw std::runtime_error("accept() failed");

        int nodelay = 1;
        setsockopt(client_, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&nodelay), sizeof(nodelay));
    }

    // The payload_size field in the 17-byte header is what makes this
    // possible: after the header arrives, the receiver knows EXACTLY how
    // many more bytes complete this message, and loops recv() until it has
    // them -- regardless of how the sender's bytes happened to be split
    // across TCP segments/packets on the way here.
    bool recvMessage(wire::Header &outHeader, std::vector<uint8_t> &outPayload) {
        uint8_t hdr[wire::HEADER_SIZE];
        if (!recvAll(hdr, sizeof(hdr))) return false;
        outHeader = wire::Header::parse(hdr);
        outPayload.resize(outHeader.payloadSize);
        if (outHeader.payloadSize && !recvAll(outPayload.data(), outHeader.payloadSize)) return false;
        return true;
    }

private:
    SOCKET listenSock_ = INVALID_SOCKET;
    SOCKET client_ = INVALID_SOCKET;

    // Mirror of sendAll: a single recv() call can return fewer bytes than
    // requested even when more are coming (TCP has no message boundaries),
    // so this keeps calling recv() until `size` bytes are actually in hand.
    bool recvAll(uint8_t *data, size_t size) {
        size_t got = 0;
        while (got < size) {
            int rc = recv(client_, reinterpret_cast<char *>(data) + got,
                           static_cast<int>(size - got), 0);
            if (rc <= 0) return false; // 0 = clean close, <0 = error
            got += static_cast<size_t>(rc);
        }
        return true;
    }
};
