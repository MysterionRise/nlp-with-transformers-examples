package org.mystic.model

import java.nio.ByteBuffer

import akka.util.ByteString

case class Deal(timestamp: Long, ticker: String, price: Double, size: Int)

object Deal {

  // todo refactor in a functional way?
  def byteString2Deal(b: ByteString): Deal = {
    val bytes = b.toArray
    val messageLen = ByteBuffer.wrap(bytes, 0, 2).getShort
    val timestamp = ByteBuffer.wrap(bytes, 2, 8).getLong
    val tickerLen = ByteBuffer.wrap(bytes, 10, 2).getShort
    val ticker = new String(ByteBuffer.wrap(bytes, 12, tickerLen).compact().array(), 0, tickerLen, "ascii")
    val price = ByteBuffer.wrap(bytes, 12 + tickerLen, 8).getDouble
    val size = ByteBuffer.wrap(bytes, 20 + tickerLen, 4).getInt
    Deal(timestamp, ticker, price, size)
  }

}
