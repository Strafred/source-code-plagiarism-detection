public class Main {
    public static void main(String[] args) {
        int jarak;
        System.out.println("Miles\t\tKilometers");
        System.out.println("-------------------------------");
        jarak = 1;
        printJarakDalamMilDanKilo(jarak);
    }
    public static void printJarakDalamMilDanKilo(int jarak){
        while (jarak <= 10) {
            System.out.println(jarak + "\t\t" + jarak * 1.609);
            jarak++;
        }
    }
}
