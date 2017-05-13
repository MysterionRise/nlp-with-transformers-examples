package org.mystic

import java.time.Instant
import java.time.temporal.ChronoUnit

import akka.actor.ActorSystem
import akka.pattern.ask
import akka.testkit.{DefaultTimeout, ImplicitSender, TestActorRef, TestKit}
import org.mystic.actors.StorageActor
import org.mystic.model._
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}

class Check10MinutesSimpleData extends TestKit(ActorSystem("Check1MinData"))
  with DefaultTimeout with ImplicitSender
  with WordSpecLike with Matchers with BeforeAndAfterAll {

  val actorRef = TestActorRef(new StorageActor)

  private val tradeTime1 = Instant.now().minus(1, ChronoUnit.MINUTES)
  private val tradeTime2 = Instant.now().minus(2, ChronoUnit.MINUTES)
  private val tradeTime3 = Instant.now().minus(3, ChronoUnit.MINUTES)
  private val tradeTime4 = Instant.now().minus(4, ChronoUnit.MINUTES)
  private val tradeTime5 = Instant.now().minus(5, ChronoUnit.MINUTES)
  private val tradeTime6 = Instant.now().minus(6, ChronoUnit.MINUTES)
  private val tradeTime7 = Instant.now().minus(7, ChronoUnit.MINUTES)
  private val tradeTime8 = Instant.now().minus(8, ChronoUnit.MINUTES)
  private val tradeTime9 = Instant.now().minus(9, ChronoUnit.MINUTES)
  private val tradeTime0 = Instant.now().minus(10, ChronoUnit.MINUTES)
  private val wrongTradeTime = Instant.now().minus(11, ChronoUnit.MINUTES)

  private val time1 = tradeTime1.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time2 = tradeTime2.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time3 = tradeTime3.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time4 = tradeTime4.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time5 = tradeTime5.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time6 = tradeTime6.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time7 = tradeTime7.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time8 = tradeTime8.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time9 = tradeTime9.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val time0 = tradeTime0.truncatedTo(ChronoUnit.MINUTES).toEpochMilli
  private val wrongTime = wrongTradeTime.truncatedTo(ChronoUnit.MINUTES).toEpochMilli


  actorRef ? Deal(tradeTime1.toEpochMilli, "A", 100.0d, 1)
  actorRef ? Deal(tradeTime2.toEpochMilli, "A", 200.0d, 2)
  actorRef ? Deal(tradeTime3.toEpochMilli, "A", 300.0d, 3)
  actorRef ? Deal(tradeTime4.toEpochMilli, "A", 400.0d, 4)
  actorRef ? Deal(tradeTime5.toEpochMilli, "A", 500.0d, 5)
  actorRef ? Deal(tradeTime6.toEpochMilli, "A", 600.0d, 6)
  actorRef ? Deal(tradeTime7.toEpochMilli, "A", 700.0d, 7)
  actorRef ? Deal(tradeTime8.toEpochMilli, "A", 800.0d, 8)
  actorRef ? Deal(tradeTime9.toEpochMilli, "A", 900.0d, 9)
  actorRef ? Deal(tradeTime0.toEpochMilli, "A", 1000.0d, 10)
  actorRef ? Deal(wrongTradeTime.toEpochMilli, "A", 999.0d, 999)

  val future = actorRef ? AskFor10MData
  val res = future.value.get.get.asInstanceOf[Data]
  res.data should have length 10

  res.data should contain(Some(CandleDeal("A", time1, 100.0d, 100.0d, 100.0d, 100.0d, 1)))
  res.data should contain(Some(CandleDeal("A", time2, 200.0d, 200.0d, 200.0d, 200.0d, 2)))
  res.data should contain(Some(CandleDeal("A", time3, 300.0d, 300.0d, 300.0d, 300.0d, 3)))
  res.data should contain(Some(CandleDeal("A", time4, 400.0d, 400.0d, 400.0d, 400.0d, 4)))
  res.data should contain(Some(CandleDeal("A", time5, 500.0d, 500.0d, 500.0d, 500.0d, 5)))
  res.data should contain(Some(CandleDeal("A", time6, 600.0d, 600.0d, 600.0d, 600.0d, 6)))
  res.data should contain(Some(CandleDeal("A", time7, 700.0d, 700.0d, 700.0d, 700.0d, 7)))
  res.data should contain(Some(CandleDeal("A", time8, 800.0d, 800.0d, 800.0d, 800.0d, 8)))
  res.data should contain(Some(CandleDeal("A", time9, 900.0d, 900.0d, 900.0d, 900.0d, 9)))
  res.data should contain(Some(CandleDeal("A", time0, 1000.0d, 1000.0d, 1000.0d, 1000.0d, 10)))

  res.data should not contain (Some(CandleDeal("A", wrongTime, 999.0d, 999.0d, 999.0d, 999.0d, 999)))


}
