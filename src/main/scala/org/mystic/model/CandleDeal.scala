package org.mystic.model

import java.time.Instant

import org.json4s.JsonDSL._
import org.json4s.native.JsonMethods._

case class CandleDeal(ticker: String, timestamp: Long, open: Double, high: Double, low: Double, close: Double, volume: Long) {

  def updateCandleWithDeal(d: Deal) = {
    new CandleDeal(ticker, timestamp, d.price, Math.max(high, d.price), Math.min(low, d.price), d.price, volume + d.size)
  }

  def toJson(): String = {
    val json =
      ("ticker" -> ticker) ~
        ("timestamp" -> Instant.ofEpochMilli(timestamp).toString) ~
        ("open" -> open) ~
        ("high" -> high) ~
        ("low" -> low) ~
        ("close" -> close) ~
        ("volume" -> volume)

    compact(render(json)) + "\n"
  }

}
