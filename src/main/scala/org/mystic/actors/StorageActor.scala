package org.mystic.actors

import java.time.temporal.ChronoUnit
import java.time.{Instant, LocalDateTime, ZoneOffset}

import akka.actor.{Actor, ActorLogging}
import org.mystic.model._

import scala.collection.mutable

class StorageActor extends Actor with ActorLogging {

  private val dealsByName = new mutable.HashMap[String, mutable.Map[Long, CandleDeal]]()

  private val MINUTE = 60000

  def truncateToMinute(timestamp: Long): Long = {
    val truncatedTimestamp = Instant.ofEpochMilli(timestamp).truncatedTo(ChronoUnit.MINUTES).toEpochMilli
    truncatedTimestamp
  }

  override def receive: Receive = {

    case d: Deal =>
      log.info(s"we got $d to store")
      val namedMap = dealsByName.getOrElse(d.ticker, new mutable.HashMap[Long, CandleDeal]())
      val candle = namedMap.getOrElse(truncateToMinute(d.timestamp),
        CandleDeal(d.ticker, truncateToMinute(d.timestamp), d.price, 0, Double.MaxValue, 0, 0))
      val updatedCandle = candle.updateCandleWithDeal(d)
      namedMap.put(truncateToMinute(d.timestamp), updatedCandle)
      dealsByName.put(d.ticker, namedMap)

    case AskFor1MData =>
      val currentMin = truncateToMinute(Instant.now().toEpochMilli)
      val candles = dealsByName.keySet.map(name => {
        dealsByName(name).get(currentMin - MINUTE)
      })
      sender() ! Data(candles.toList)

    case AskFor10MData =>
      val currentMin = truncateToMinute(Instant.now().toEpochMilli)
      val crossProduct = for {
        x <- 10 to 1 by -1
        y <- dealsByName.keySet
      } yield (x, y)
      val candles = crossProduct.map(x => {
        dealsByName(x._2).get(currentMin - x._1 * MINUTE)
      })
      sender() ! Data(candles.toList)

    case _ =>
      log.error("something goes wrong in storage actor")

  }
}
